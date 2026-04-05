"""
Simplified web interface for spatial transcriptomics analysis.

Uses AnnData + image instead of spatialdata.
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from pathlib import Path
import logging
import io
import base64
import os
import asyncio
import threading
from datetime import datetime
from typing import Optional, List, Dict
from PIL import Image
import anndata as ad
import numpy as np
import matplotlib
matplotlib.use('Agg')

from stat_agent.core.session import SimpleSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import agent (optional - falls back to rule-based if not available)
try:
    from stat_agent.agent.spatial_agent_core import SpatialAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent module not available: {e}")
    logger.warning("Using rule-based responses only")
    AGENT_AVAILABLE = False
    SpatialAgent = None

# Create Flask app
_WEB_DIR = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(_WEB_DIR / "templates"),
    static_folder=str(_WEB_DIR / "static"),
)
app.config['SECRET_KEY'] = 'spatial-agent-secret-key'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Global state
session: SimpleSession = None
agent: Optional[SpatialAgent] = None  # LLM-powered agent (optional)
llm_config: Optional[Dict] = None  # Store LLM config for reuse in skills
chat_abort_event = threading.Event()  # Signal to abort current chat processing
chat_busy = False  # True while agent is processing a request
_pending_visual_events = []  # Visual events from clarification requests (no turn created yet)


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/skills')
def skills():
    """Serve the skills catalog page."""
    return render_template('skills.html')


@app.route('/guide')
def guide():
    """Serve the getting started guide page."""
    return render_template('guide.html')


@app.route('/api/init_dataset', methods=['POST'])
def initialize_dataset():
    """Initialize session with multi-format dataset (new loader)."""
    global session, agent, llm_config

    data = request.json
    dataset_dir = data.get('dataset_dir')
    session_name = data.get('session_name', 'web_session')

    # Optional API configuration
    user_api_key = data.get('api_key')
    user_model = data.get('model')
    user_base_url = data.get('base_url')

    try:
        # Create session
        logger.info(f"Creating session: {session_name}")
        session = SimpleSession(name=session_name)

        # Load dataset (auto-detects format)
        logger.info(f"Loading dataset from: {dataset_dir}")
        session.load_dataset(dataset_dir)

        # Initialize agent if available
        if AGENT_AVAILABLE:
            api_key = user_api_key or os.getenv("POE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            model = user_model or os.getenv("SPATIAL_AGENT_MODEL", "gpt-4o")

            if api_key:
                try:
                    logger.info(f"Initializing agent with model: {model}")
                    agent_kwargs = {
                        "model": model,
                        "api_key": api_key,
                        "session": session,
                        "enable_planning": True,
                        "enable_skills": True,
                        # skill_dir will auto-discover .claude/skills/
                    }

                    if user_base_url:
                        agent_kwargs["endpoint"] = user_base_url

                    agent = SpatialAgent(**agent_kwargs)
                    logger.info("Agent initialized successfully")

                    # Create LLM config for session and notebook
                    llm_config = {
                        'api_key': api_key,
                        'model': model,
                        'base_url': user_base_url
                    }
                    # Store in session for skill access
                    session.llm_config = llm_config
                    logger.info(f"Stored LLM config for skill reuse: model={model}")

                    # Initialize notebook for this session (with LLM config)
                    agent.notebook_logger.initialize_notebook(
                        dataset_path=dataset_dir,
                        llm_config=llm_config
                    )
                    logger.info("Notebook logging initialized with LLM config")

                    # Coordinate prompt logger to use same session directory
                    agent.prompt_logger.set_session_dir(agent.notebook_logger.get_session_dir())
                    logger.info("Prompt logger coordinated with notebook logger")

                except Exception as e:
                    logger.warning(f"Failed to initialize agent: {e}")
                    agent = None
                    llm_config = None
            else:
                agent = None
                llm_config = None
        else:
            llm_config = None

        # Get summary
        summary = session.get_frontend_summary()

        return jsonify({
            'success': True,
            'message': f'Dataset loaded successfully ({summary.get("data_format")})',
            'summary': summary,
            'agent_active': agent is not None
        })

    except Exception as e:
        logger.error(f"Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/select_slice', methods=['POST'])
def api_select_slice():
    """Select a specific slice for multi-slice data."""
    global session

    if session is None:
        return jsonify({
            'success': False,
            'error': 'Session not initialized'
        }), 400

    data = request.json
    slice_id = data.get('slice_id')

    if slice_id is None:
        return jsonify({
            'success': False,
            'error': 'slice_id required'
        }), 400

    try:
        # Validate slice exists
        if slice_id not in session.slices:
            available = session.slice_ids
            return jsonify({
                'success': False,
                'error': f'Slice {slice_id} not found. Available: {available}'
            }), 400

        # Set current slice (simple assignment)
        session.current_slice_id = slice_id
        summary = session.get_frontend_summary()

        return jsonify({
            'success': True,
            'message': f'Selected slice {slice_id}',
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Slice selection failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/select_modality', methods=['POST'])
def api_select_modality():
    """Select gene or protein modality."""
    global session

    if session is None:
        return jsonify({
            'success': False,
            'error': 'Session not initialized'
        }), 400

    data = request.json
    modality = data.get('modality')

    if modality not in ['gene', 'protein']:
        return jsonify({
            'success': False,
            'error': 'modality must be "gene" or "protein"'
        }), 400

    try:
        # Find slices with requested modality
        matching_slices = [s for s in session.slices.values() if s.modality == modality]

        if not matching_slices:
            return jsonify({
                'success': False,
                'error': f'No slices found with {modality} modality'
            }), 404

        # Select first matching slice (or keep current if it matches)
        if session.current_slice_id is not None:
            current = session.current_slice()
            if current and current.modality == modality:
                # Already on correct modality
                target_slice = current
            else:
                target_slice = matching_slices[0]
        else:
            target_slice = matching_slices[0]

        session.current_slice_id = target_slice.slice_id
        summary = session.get_frontend_summary()

        return jsonify({
            'success': True,
            'message': f'Selected {modality} modality (slice {target_slice.slice_id})',
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Modality selection failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/session/summary', methods=['GET'])
def get_session_summary():
    """Get current session summary."""
    global session

    if session is None:
        return jsonify({
            'success': False,
            'error': 'Session not initialized'
        }), 400

    try:
        summary = session.get_frontend_summary()
        return jsonify({
            'success': True,
            'summary': summary,
            'agent_active': agent is not None,
            'chat_busy': chat_busy,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/notebook/url', methods=['GET'])
def get_notebook_url():
    """Get Jupyter Lab URL for the current session's notebook."""
    global agent

    if agent is None or not agent.notebook_logger.initialized:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    notebook_file = agent.notebook_logger.notebook_file
    if notebook_file is None or not notebook_file.exists():
        return jsonify({'success': False, 'error': 'Notebook not found'}), 404

    # Relative path from project root for Jupyter URL
    try:
        rel_path = notebook_file.relative_to(Path.cwd())
    except ValueError:
        rel_path = notebook_file

    jupyter_port = app.config.get('JUPYTER_PORT', 8890)
    return jsonify({
        'success': True,
        'jupyter_port': jupyter_port,
        'notebook_path': str(rel_path),
    })


@app.route('/api/image/data', methods=['GET'])
def get_image_data():
    """
    Get image data for display.

    Query parameters:
    - slice_id (optional): Specific slice ID for multi-slice data
    - modality (optional): 'gene' or 'protein' for multi-omics data

    If not provided, returns current slice/modality.
    """
    global session

    if session is None or not session.has_data:
        return jsonify({
            'success': False,
            'error': 'No data loaded'
        }), 400

    try:
        # Get optional parameters
        requested_slice_id = request.args.get('slice_id', type=int)

        # Determine which slice to return
        if requested_slice_id is not None:
            # Specific slice requested
            if requested_slice_id not in session.slices:
                return jsonify({
                    'success': False,
                    'error': f'Slice {requested_slice_id} not found. Available: {session.slice_ids}'
                }), 404
            target_slice = session.get_slice(requested_slice_id)
        else:
            # Use current slice
            target_slice = session.current_slice()
            if target_slice is None:
                return jsonify({
                    'success': False,
                    'error': 'No current slice selected'
                }), 400

        # Get image from slice
        image = target_slice.primary_image

        # Handle missing image: generate cell scatter plot
        if image is None:
            logger.info("No image available. Generating cell scatter plot fallback.")
            import time
            start_time = time.time()

            # Generate scatter plot from adata
            image = _generate_cell_scatter_image(target_slice.adata)

            logger.info(f"Scatter plot generated in {time.time()-start_time:.2f}s")

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            # Normalize to 0-255
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(image, mode='L')

        # OPTIMIZATION: Use JPEG for faster transfer (much smaller than PNG)
        # For large scatter plots, use lower quality to reduce file size
        buf = io.BytesIO()
        jpeg_quality = 75  # Lower quality for large images
        pil_img.save(buf, format='JPEG', quality=jpeg_quality, optimize=True)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Image dimensions (height, width) -> (width, height)
        img_width = int(image.shape[1])
        img_height = int(image.shape[0])

        # Calculate transfer size
        transfer_size_mb = len(img_base64) / (1024 * 1024)
        logger.info(f"Sending image: {img_width}x{img_height}, size: {transfer_size_mb:.2f} MB")

        return jsonify({
            'success': True,
            'image_data': img_base64,
            'image_width': img_width,
            'image_height': img_height,
            'is_scatter': target_slice.primary_image is None,
            'slice_id': target_slice.slice_id,
            'modality': target_slice.modality,
            'note': 'Full resolution image. Cell at (x,y) maps to image pixel (x,y).'
        })

    except Exception as e:
        logger.error(f"Failed to get image data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _generate_cell_scatter_image(adata: ad.AnnData) -> np.ndarray:
    """
    Generate a blank background image when no tissue image is available.

    CRITICAL: Maintains 1:1 coordinate mapping - Cell/Spot at (x, y) = Image pixel (x, y)
    This ensures cell overlay and ROI selection work correctly.

    Cell visualization is handled entirely by the overlay system (controlled by
    Cell Type Selection checkboxes), not by this background image.

    Parameters
    ----------
    adata : AnnData
        AnnData object with cell/spot coordinates

    Returns
    -------
    image : ndarray
        White RGB image sized to match coordinate space
    """
    # Get coordinates
    x = adata.obs['x'].values
    y = adata.obs['y'].values

    # Filter out NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.warning(f"Filtering {n_invalid} entries with NaN coordinates")
        x = x[valid_mask]
        y = y[valid_mask]

    # CRITICAL: Image dimensions must match coordinate space for 1:1 mapping
    x_max = np.ceil(x.max()).astype(int)
    y_max = np.ceil(y.max()).astype(int)

    img_width = x_max + 1   # +1 to include the max coordinate
    img_height = y_max + 1

    logger.info(f"Generating blank background: {img_width}x{img_height}")

    # Plain white background - cell visualization handled by overlay system
    image = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    return image


@app.route('/api/roi/add', methods=['POST'])
def add_roi():
    """Add a new ROI and extract cells."""
    global session, agent

    if session is None or not session.has_data:
        return jsonify({
            'success': False,
            'error': 'Session not initialized'
        }), 400

    data = request.json
    roi_name = data.get('name')
    roi_type = data.get('type')
    params = data.get('params')

    try:
        if roi_type == 'bbox':
            min_x = params['min_x']
            min_y = params['min_y']
            max_x = params['max_x']
            max_y = params['max_y']

            # CRITICAL: ROI belongs to current slice
            current_slice_id = session.current_slice_id
            current_slice = session.current_slice()
            if current_slice is None:
                return jsonify({
                    'success': False,
                    'error': 'No current slice selected'
                }), 400
            current_modality = current_slice.modality

            # Debug logging
            logger.info(f"Creating ROI '{roi_name}' with slice_id={current_slice_id}, modality={current_modality}")

            # Create ROI definition and filter cells
            roi_definition = {
                'type': 'bbox',
                'x_min': min_x,
                'x_max': max_x,
                'y_min': min_y,
                'y_max': max_y
            }

            # Create ROI for this slice (filters and stores data)
            session.create_roi(roi_name, current_slice_id, roi_definition)

            # Get the ROI object (consistent API)
            roi = session.get_roi(roi_name)
            if roi is None:
                return jsonify({
                    'success': False,
                    'error': f'Failed to create ROI {roi_name}'
                }), 500

            # Get cell type distribution if available (convert to native Python types)
            celltype_dist = {}
            if 'celltype' in roi.adata.obs.columns:
                celltype_counts = roi.adata.obs['celltype'].value_counts()
                celltype_dist = {k: int(v) for k, v in celltype_counts.to_dict().items()}

            # Track ROI creation in agent memory (for context awareness)
            if agent is not None:
                agent.memory.track_roi_created(
                    roi_name,
                    metadata={
                        'n_cells': roi.n_obs,
                        'slice_id': current_slice_id,
                        'modality': current_modality
                    }
                )

                # Log ROI creation to notebook
                agent.notebook_logger.append_roi_creation(
                    roi_name=roi_name,
                    slice_id=current_slice_id,
                    roi_definition=roi_definition
                )

            return jsonify({
                'success': True,
                'roi_name': roi_name,
                'n_cells': roi.n_obs,
                'celltype_distribution': celltype_dist,
                'bounds': [float(min_x), float(min_y), float(max_x), float(max_y)],
                'slice_id': current_slice_id,
                'modality': current_modality
            })

        elif roi_type == 'polygon':
            vertices = params.get('vertices', [])
            if len(vertices) < 3:
                return jsonify({
                    'success': False,
                    'error': 'Polygon requires at least 3 vertices'
                }), 400

            # Convert to list of tuples
            vertices_tuples = [(v[0], v[1]) for v in vertices]

            current_slice_id = session.current_slice_id
            current_slice = session.current_slice()
            if current_slice is None:
                return jsonify({
                    'success': False,
                    'error': 'No current slice selected'
                }), 400
            current_modality = current_slice.modality

            logger.info(f"Creating polygon ROI '{roi_name}' with {len(vertices)} vertices, slice_id={current_slice_id}")

            roi_definition = {
                'type': 'polygon',
                'vertices': vertices_tuples
            }

            session.create_roi(roi_name, current_slice_id, roi_definition)

            roi = session.get_roi(roi_name)
            if roi is None:
                return jsonify({
                    'success': False,
                    'error': f'Failed to create ROI {roi_name}'
                }), 500

            celltype_dist = {}
            if 'celltype' in roi.adata.obs.columns:
                celltype_counts = roi.adata.obs['celltype'].value_counts()
                celltype_dist = {k: int(v) for k, v in celltype_counts.to_dict().items()}

            if agent is not None:
                agent.memory.track_roi_created(
                    roi_name,
                    metadata={
                        'n_cells': roi.n_obs,
                        'slice_id': current_slice_id,
                        'modality': current_modality,
                        'type': 'polygon',
                        'n_vertices': len(vertices)
                    }
                )
                agent.notebook_logger.append_roi_creation(
                    roi_name=roi_name,
                    slice_id=current_slice_id,
                    roi_definition=roi_definition
                )

            bounds = roi.bounds
            return jsonify({
                'success': True,
                'roi_name': roi_name,
                'n_cells': roi.n_obs,
                'celltype_distribution': celltype_dist,
                'bounds': [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
                'vertices': vertices,
                'slice_id': current_slice_id,
                'modality': current_modality
            })

        else:
            return jsonify({
                'success': False,
                'error': f'ROI type "{roi_type}" not yet supported. Use "bbox" or "polygon" (freehand).'
            }), 400

    except Exception as e:
        logger.error(f"Failed to add ROI: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/roi/list', methods=['GET'])
def list_rois():
    """List all ROIs."""
    global session

    if session is None:
        return jsonify({
            'success': False,
            'error': 'Session not initialized'
        }), 400

    try:
        # Get all ROIs from the manager
        rois = list(session.roi_manager.rois.values())
        roi_data = []

        for roi in rois:
            # Convert bounds to native Python types
            bounds = roi.bounds
            if bounds is not None:
                bounds = [float(x) for x in bounds]

            roi_info = {
                'name': roi.name,
                'type': roi.type,
                'bounds': bounds,
                'slice_id': roi.slice_id,  # Include slice_id
                'modality': roi.modality   # Include modality
            }

            # Include vertices for polygon ROIs
            if roi.type == 'polygon' and hasattr(roi.geometry, 'exterior'):
                coords = list(roi.geometry.exterior.coords)
                # Shapely includes the closing point (same as first), drop it
                if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
                roi_info['vertices'] = [[float(x), float(y)] for x, y in coords]

            # Add cell count if available (simplified structure - no nesting)
            if roi.name in session.roi_subsets:
                roi_info['n_cells'] = int(session.roi_subsets[roi.name].n_obs)

            roi_data.append(roi_info)

        return jsonify({
            'success': True,
            'rois': roi_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/roi/delete', methods=['POST'])
def delete_roi():
    """Delete an ROI by name."""
    global session

    if session is None:
        return jsonify({'success': False, 'error': 'Session not initialized'}), 400

    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'error': 'ROI name required'}), 400

    try:
        if name in session.roi_manager.rois:
            session.roi_manager.remove_roi(name)
            if name in session.roi_subsets:
                del session.roi_subsets[name]
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': f'ROI "{name}" not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/roi/rename', methods=['POST'])
def rename_roi():
    """Rename an ROI."""
    global session

    if session is None:
        return jsonify({'success': False, 'error': 'Session not initialized'}), 400

    data = request.json
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    if not old_name or not new_name:
        return jsonify({'success': False, 'error': 'old_name and new_name required'}), 400
    if new_name in session.roi_manager.rois:
        return jsonify({'success': False, 'error': f'ROI "{new_name}" already exists'}), 400

    try:
        if old_name not in session.roi_manager.rois:
            return jsonify({'success': False, 'error': f'ROI "{old_name}" not found'}), 404
        roi = session.roi_manager.rois.pop(old_name)
        roi.name = new_name
        session.roi_manager.rois[new_name] = roi
        if old_name in session.roi_subsets:
            session.roi_subsets[new_name] = session.roi_subsets.pop(old_name)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cells/overlay', methods=['GET'])
def get_cell_overlay():
    """Get cell positions for overlay on image (always loads all cells).

    Optional parameters:
    - slice_id: Specific slice to load (for multi-slice data)
    - modality: Specific modality to load (for multi-omics data)
    - selected_celltypes: Comma-separated list of celltypes to filter
    """
    global session

    if session is None or not session.has_data:
        return jsonify({
            'success': False,
            'error': 'No data loaded'
        }), 400

    try:
        # Get optional parameters
        requested_slice_id = request.args.get('slice_id', type=int)
        requested_modality = request.args.get('modality', type=str)
        selected_celltypes_param = request.args.get('selected_celltypes')
        selected_celltypes = None
        if selected_celltypes_param:
            selected_celltypes = [ct.strip() for ct in selected_celltypes_param.split(',')]

        # Determine which slice to use
        requested_slice_id = request.args.get('slice_id', type=int)

        if requested_slice_id is not None:
            # Specific slice requested
            if requested_slice_id not in session.slices:
                return jsonify({
                    'success': False,
                    'error': f'Slice {requested_slice_id} not found'
                }), 404
            target_slice = session.get_slice(requested_slice_id)
        else:
            # Use current slice
            target_slice = session.current_slice()
            if target_slice is None:
                return jsonify({
                    'success': False,
                    'error': 'No current slice selected'
                }), 400

        target_adata = target_slice.adata

        # Always load all cells (no sampling)
        cells = target_adata.obs

        # Filter out NaN coordinates (data quality)
        valid_mask = ~(cells['x'].isna() | cells['y'].isna())
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"Filtering {n_invalid} entries with NaN coordinates")
            cells = cells[valid_mask]

        # Filter by selected celltypes if specified
        if selected_celltypes and 'celltype' in cells.columns:
            cells = cells[cells['celltype'].isin(selected_celltypes)]
            logger.info(f"Filtered to {len(cells)} cells from celltypes: {selected_celltypes}")

        # Get positions
        cell_data = {
            'x': cells['x'].tolist(),
            'y': cells['y'].tolist(),
        }

        # Detect spot data
        is_spot_data = 'spot_shape' in target_adata.uns and 'spot_diameter' in target_adata.uns
        spot_info = None

        if is_spot_data:
            # Convert numpy types to native Python types for JSON serialization
            spot_info = {
                'spot_shape': str(target_adata.uns['spot_shape']),
                'spot_diameter': int(target_adata.uns['spot_diameter']),
                'has_deconv_weights': bool(target_adata.uns.get('has_deconv_weights', False))
            }

            # Add deconvolution weights for pie chart mode
            if 'deconv_weights' in target_adata.obsm:
                # Get deconv weights for filtered cells
                deconv_weights_df = target_adata.obsm['deconv_weights'].loc[cells.index]

                # Convert to dict format: {celltype: [values...]}
                deconv_weights_dict = {
                    col: deconv_weights_df[col].tolist()
                    for col in deconv_weights_df.columns
                }
                spot_info['deconv_weights'] = deconv_weights_dict
                spot_info['celltype_order'] = list(deconv_weights_df.columns)

                logger.info(f"Included deconv_weights for {len(deconv_weights_df)} spots, {len(deconv_weights_df.columns)} celltypes")

        # Add celltype and use shared colors if available
        celltype_list = None
        celltype_colors = None
        has_celltype = 'celltype' in target_adata.obs.columns

        if has_celltype:
            cell_data['celltype'] = cells['celltype'].tolist()

            # Get unique cell types from target slice
            current_slice_celltypes = sorted(target_adata.obs['celltype'].unique().tolist())
            # Filter out NaN values
            current_slice_celltypes = [ct for ct in current_slice_celltypes if isinstance(ct, str)]
            celltype_list = current_slice_celltypes

            # Priority 1: Use per-slice colors (takes priority after annotation)
            # Priority 2: Use global shared colors (ONLY if they cover all celltypes)
            # Priority 3: Generate new colors for current slice
            if 'celltype_colors' in target_adata.uns:
                # Use target slice's own colors (set during annotation)
                celltype_colors = target_adata.uns['celltype_colors']
                logger.info(f"Using per-slice colors for {len(celltype_colors)} celltypes")
            elif session.celltype_colors and all(ct in session.celltype_colors for ct in current_slice_celltypes):
                # Use session-level shared colors ONLY if they cover all celltypes
                celltype_colors = session.celltype_colors
                logger.info(f"Using shared colors (all {len(current_slice_celltypes)} celltypes covered)")
            else:
                # Generate new colors for target slice
                logger.info(f"Generating new colors for target slice ({len(current_slice_celltypes)} celltypes)")
                colors = _generate_color_palette(len(current_slice_celltypes))
                target_adata.uns['celltype_colors'] = {
                    ct: colors[i] for i, ct in enumerate(current_slice_celltypes)
                }
                celltype_colors = target_adata.uns['celltype_colors']
        else:
            logger.info("No celltype annotations available. Cells will be displayed without type information.")

        return jsonify({
            'success': True,
            'cells': cell_data,
            'n_cells': int(len(cells)),  # Filtered cells count
            'total_cells': int(target_slice.n_obs),  # Total cells in this slice
            'has_celltype': bool(has_celltype),
            'celltypes': celltype_list,
            'celltype_colors': celltype_colors,
            'is_spot_data': bool(is_spot_data),
            'spot_info': spot_info,  # NEW: spot properties and deconv_weights
            'message': None if has_celltype else 'No celltype annotations available. You can annotate celltypes using a reference dataset.'
        })

    except Exception as e:
        logger.error(f"Failed to get cell overlay: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _generate_color_palette(n):
    """
    Generate n distinct colors using HSL color space.

    Returns dict mapping index to RGB color string.
    """
    colors = {}
    golden_ratio = 0.618033988749895
    hue = np.random.random()  # Start with random hue

    for i in range(n):
        hue += golden_ratio
        hue %= 1
        saturation = 0.6 + np.random.random() * 0.2  # 0.6-0.8
        lightness = 0.45 + np.random.random() * 0.15  # 0.45-0.6

        # Convert HSL to RGB
        r, g, b = _hsl_to_rgb(hue, saturation, lightness)
        colors[i] = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

    return colors


def _hsl_to_rgb(h, s, l):
    """Convert HSL to RGB."""
    if s == 0:
        r = g = b = l
    else:
        def hue2rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)

    return r, g, b


@app.route('/api/cells/gene_expression', methods=['GET'])
def get_gene_expression():
    """Get gene expression for spatial visualization (always loads all cells)."""
    global session

    if session is None or not session.has_data:
        return jsonify({
            'success': False,
            'error': 'No data loaded'
        }), 400

    gene = request.args.get('gene')
    if not gene:
        return jsonify({
            'success': False,
            'error': 'Gene name is required'
        }), 400

    try:
        # Get current slice
        current = session.current_slice()
        if current is None:
            return jsonify({
                'success': False,
                'error': 'No current slice selected'
            }), 400

        # Check if gene exists
        if gene not in current.adata.var_names:
            return jsonify({
                'success': False,
                'error': f'Gene "{gene}" not found in dataset'
            }), 404

        # Always load all cells (no sampling)
        cells = current.adata.obs
        expression = current.adata[:, gene].X.toarray().flatten()

        # Get positions and expression
        cell_data = {
            'x': cells['x'].tolist(),
            'y': cells['y'].tolist(),
            'expression': expression.tolist()
        }

        # Get expression range for color mapping
        expr_min = float(expression.min())
        expr_max = float(expression.max())

        return jsonify({
            'success': True,
            'gene': gene,
            'cells': cell_data,
            'n_cells': len(cells),
            'total_cells': current.n_obs,
            'expression_range': [expr_min, expr_max]
        })

    except Exception as e:
        logger.error(f"Failed to get gene expression: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/genes/list', methods=['GET'])
def list_genes():
    """List all available genes."""
    global session

    if session is None or not session.has_data:
        return jsonify({
            'success': False,
            'error': 'No data loaded'
        }), 400

    try:
        # Get current slice
        current = session.current_slice()
        if current is None:
            return jsonify({
                'success': False,
                'error': 'No current slice selected'
            }), 400

        genes = current.adata.var_names.tolist()
        return jsonify({
            'success': True,
            'genes': genes,
            'n_genes': len(genes)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the agent interface."""
    global session, agent

    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({
            'success': False,
            'error': 'Empty message'
        }), 400

    try:
        # Use LLM-powered agent if available
        if agent is not None:
            logger.info("Using LLM-powered agent for response")
            try:
                # Run async agent in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response_text = loop.run_until_complete(
                    agent.chat(user_message, execute_code=True, stream=False)
                )
                loop.close()

                # Get plots from last execution
                plots = getattr(agent, '_last_plots', [])

                # Get comprehensive state changes from agent
                state_changes = agent.get_last_state_changes()
                logger.info(f"State changes detected: {state_changes}")

                # Check if any celltypes or deconv_weights were updated
                celltype_updated = len(state_changes.get('celltypes_updated', [])) > 0
                deconv_updated = len(state_changes.get('deconv_weights_updated', [])) > 0

                return jsonify({
                    'success': True,
                    'message': response_text,
                    'plots': plots if plots else None,
                    'agent_powered': True,
                    'state_changes': state_changes,  # NEW: Comprehensive state changes
                    # Legacy fields for backward compatibility
                    'celltype_updated': celltype_updated,
                    'deconv_weights_updated': deconv_updated
                })
            except Exception as e:
                logger.error(f"Agent error: {e}, falling back to rule-based")
                import traceback
                traceback.print_exc()
                # Fall through to rule-based

        # Fall back to simple rule-based responses
        logger.info("Using rule-based responses")
        response = _handle_chat_query(user_message, session)

        return jsonify({
            'success': True,
            'message': response['message'],
            'code': response.get('code'),
            'data': response.get('data'),
            'agent_powered': False,
            'celltype_updated': False
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/abort', methods=['POST'])
def chat_abort():
    """Signal the current chat processing to stop."""
    chat_abort_event.set()
    # Clear clarification state so next query starts fresh
    if agent is not None:
        try:
            agent._clear_clarification_context()
        except Exception:
            pass
    logger.info("Chat aborted by user")
    return jsonify({'success': True})


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle chat messages with real-time streaming progress updates."""
    global session, agent

    # If previous request was aborted, clear stale clarification state
    # (old background threads may have set clarification state after abort)
    if chat_abort_event.is_set() and agent is not None:
        try:
            agent._clear_clarification_context()
            logger.info("Cleared stale clarification context from previous abort")
        except Exception:
            pass

    # Clear abort signal for this new request
    chat_abort_event.clear()

    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({
            'success': False,
            'error': 'Empty message'
        }), 400

    def generate():
        """Generator function for Server-Sent Events."""
        global chat_busy
        import json
        import time

        chat_busy = True
        try:
            # Removed initial progress messages - only show clarification/prerequisites blocks
            # yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...'})}\n\n"
            # time.sleep(0.05)

            # Use LLM-powered agent if available
            if agent is not None:
                logger.info("Using LLM-powered agent for streaming response")
                # Detect if this is a clarification response BEFORE processing
                _is_continuation = (
                    hasattr(agent, 'clarification_context')
                    and agent.clarification_context
                    and agent.clarification_context.is_waiting_for_clarification()
                )
                # Capture the original query before clarification context is cleared.
                # For planner clarifications, it's stored in _original_query.
                # For skill_selection/verifier, look back in memory messages for the
                # last user message before this one (which is the original query).
                _original_query = None
                if _is_continuation:
                    # Try clarification context first
                    _original_query = getattr(agent.clarification_context, '_original_query', None)
                    # Fall back: find the previous user message in memory
                    if not _original_query and agent.memory:
                        user_msgs = [m for m in agent.memory.messages if m.role == 'user']
                        if user_msgs:
                            _original_query = user_msgs[-1].content
                logger.info(f"Chat stream: is_continuation={_is_continuation}, original_query={_original_query[:80] if _original_query else None}")
                # Track turn count to detect if a new turn was created
                _turns_before = len(agent.memory.turns) if agent and agent.memory else 0

                # Removed thinking/matching status - only show clarification/prerequisites blocks
                # yield f"data: {json.dumps({'type': 'progress', 'message': '🤔 Analyzing your request...'})}\n\n"
                # time.sleep(0.1)
                # yield f"data: {json.dumps({'type': 'progress', 'message': '🔍 Matching appropriate analysis methods...'})}\n\n"
                # time.sleep(0.1)

                try:
                    # Run async agent with event streaming using queue for real-time updates
                    import queue
                    import threading

                    event_queue = queue.Queue()
                    final_plots = []
                    final_state_changes = None
                    is_multi_step_plan = False  # Track if this is a multi-step plan

                    # Async event processing in separate thread
                    async def stream_events():
                        nonlocal final_plots, final_state_changes, is_multi_step_plan

                        logger.info("stream_events: Starting async iteration over agent.chat_with_events()...")
                        event_counter = 0

                        # Map pipeline events to human-readable log messages for the detail panel
                        _pipeline_log_map = {
                            'pipeline_start': lambda e: f"Processing: {e.get('query', '')[:80]}",
                            'planning_start': lambda e: "Planning analysis steps...",
                            'planning_complete': lambda e: f"Plan ready ({e.get('steps', '?')} steps)",
                            'filter_complete': lambda e: f"Skill filter: {e.get('count', '?')} compatible",
                            'matching_complete': lambda e: f"Skill matching: {e.get('matched', '?')} matched",
                            'skill_matched': lambda e: f"Selected: {e.get('skill', '?')}",
                            'no_skill_matched': lambda e: f"No skill matched (step {e.get('step', '?')})",
                            'verification_start': lambda e: f"Verifying: {e.get('skill', '?')}",
                            'verification_complete': lambda e: "Prerequisites met",
                            'execution_start': lambda e: "Generating code...",
                            'step_start': lambda e: f"Step {e.get('step_number', '?')}: {e.get('description', '')}",
                            'execution_complete': lambda e: "Execution complete",
                            'step_execution_complete': lambda e: f"Step {e.get('step_number', '?')} complete",
                            'reflection_start': lambda e: "Reflecting on error...",
                            'reflection_complete': lambda e: "Reflection complete",
                            'agent_text': lambda e: "Agent response...",
                            'code_block_complete': lambda e: "Code block complete",
                        }

                        async for event in agent.chat_with_events(user_message, execute_code=True):
                            # Stop processing if user aborted
                            if chat_abort_event.is_set():
                                logger.info("stream_events: Abort detected, stopping async iteration")
                                break

                            event_counter += 1
                            event_type = event.get('type')
                            if event_type == 'execution_output':
                                logger.debug(f"stream_events: Received event #{event_counter}: type={event_type}")
                            else:
                                logger.info(f"stream_events: Received event #{event_counter}: type={event_type}")

                            # Emit pipeline_log for frontend detail panel
                            if event_type in _pipeline_log_map:
                                try:
                                    log_msg = _pipeline_log_map[event_type](event)
                                    event_queue.put({'type': 'pipeline_log', 'message': log_msg})
                                except Exception:
                                    pass  # Don't let log formatting break the pipeline

                            # Handle new pipeline event types
                            if event_type == 'pipeline_start':
                                # Just log, don't forward to frontend
                                logger.debug(f"Pipeline started for query: {event.get('query', '')[:50]}...")

                            elif event_type == 'planning_complete':
                                # Forward plan to frontend
                                logger.info(f"stream_events: Putting planning_complete into queue")
                                event_queue.put(event)

                            elif event_type == 'clarification_needed':
                                # Forward clarification request to frontend
                                logger.info(f"stream_events: Putting clarification_needed into queue")
                                event_queue.put(event)

                            elif event_type == 'skill_selection':
                                # Forward skill selection to frontend
                                logger.info(f"stream_events: Putting skill_selection into queue")
                                event_queue.put(event)

                            elif event_type == 'prerequisites_needed':
                                # Forward prerequisites request to frontend
                                logger.info(f"stream_events: Putting prerequisites_needed into queue")
                                event_queue.put(event)

                            elif event_type == 'advice':
                                # Forward advice to frontend
                                logger.info(f"stream_events: Putting advice into queue")
                                event_queue.put(event)

                            elif event_type == 'warning':
                                # Forward warning to frontend
                                logger.info(f"stream_events: Putting warning into queue")
                                event_queue.put(event)

                            elif event_type == 'execution_complete':
                                # Forward execution result to frontend
                                logger.info(f"stream_events: Putting execution_complete into queue")
                                event_queue.put(event)
                                if event.get('plots'):
                                    final_plots.extend(event['plots'])

                            # Handle old event types (for backward compatibility)
                            elif event_type == 'plan_created':
                                # Mark this as a multi-step plan
                                is_multi_step_plan = True

                                # Send plan summary to frontend
                                plan = event['plan']
                                plan_summary = f"**Plan Created** ({len(plan['steps'])} steps):\n"
                                for step in plan['steps']:
                                    plan_summary += f"{step['step_number']}. {step['description']}\n"

                                logger.info(f"stream_events: Putting plan_created into queue")
                                event_queue.put({'type': 'plan_created', 'plan': plan, 'summary': plan_summary})

                            elif event_type == 'step_start':
                                logger.info(f"stream_events: Putting step_start into queue")
                                event_queue.put(event)

                            elif event_type == 'execution_output':
                                logger.debug(f"stream_events: Putting execution_output into queue")
                                event_queue.put(event)

                            elif event_type == 'step_complete':
                                logger.info(f"stream_events: Putting step_complete into queue")
                                event_queue.put(event)
                                # Accumulate plots
                                if event.get('plots'):
                                    final_plots.extend(event['plots'])

                            elif event_type == 'state_changes':
                                # CRITICAL: Capture state changes event
                                logger.info(f"stream_events: Received state_changes event, capturing changes")
                                final_state_changes = event.get('changes', {})
                                logger.info(f"stream_events: State changes captured: {final_state_changes}")
                                # Don't forward to frontend - handled in 'done' event

                            elif event_type == 'pipeline_complete':
                                # Capture plots if present
                                logger.info(f"stream_events: Received pipeline_complete")
                                if event.get('plots'):
                                    final_plots.extend(event['plots'])
                                # Don't forward to frontend

                            elif event_type in ('reflection_start', 'reflection_complete'):
                                # Don't forward reflection events to frontend - internal error handling
                                logger.debug(f"stream_events: Skipping {event_type} event (not shown to user)")

                            elif event_type == 'agent_text':
                                # Forward agent text segment to frontend
                                logger.debug(f"stream_events: Putting agent_text into queue")
                                event_queue.put(event)

                            elif event_type == 'code_block_complete':
                                # Forward code block completion to frontend
                                logger.info(f"stream_events: Putting code_block_complete into queue")
                                event_queue.put(event)
                                # Accumulate plots from code blocks
                                if event.get('plots'):
                                    final_plots.extend(event['plots'])

                            elif event_type == 'step_execution_complete':
                                # Forward step execution complete to frontend
                                logger.info(f"stream_events: Putting step_execution_complete into queue")
                                event_queue.put(event)
                                # Accumulate plots
                                if event.get('plots'):
                                    final_plots.extend(event['plots'])

                            elif event_type == 'execution_issue':
                                # Forward execution issue to frontend
                                logger.info(f"stream_events: Putting execution_issue into queue")
                                event_queue.put(event)

                            elif event_type in (
                                'planning_start', 'filter_complete', 'matching_complete',
                                'skill_matched', 'no_skill_matched', 'verification_start',
                                'verification_complete', 'execution_start', 'pipeline_result'
                            ):
                                # Pipeline detail events - already emitted as pipeline_log above
                                logger.debug(f"stream_events: Pipeline detail event '{event_type}' (logged via pipeline_log)")

                            else:
                                # Unknown event type - forward it anyway for debugging
                                logger.warning(f"stream_events: Unknown event type '{event_type}', forwarding to frontend")
                                event_queue.put(event)

                        # Completion is signaled in finally block (no need to signal here)
                        logger.info(f"stream_events: Completed iteration, received {event_counter} events total")

                    # Run async processing in thread
                    def run_async_stream():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(stream_events())
                        except Exception as e:
                            logger.error(f"Error in async stream: {e}")
                            import traceback
                            traceback.print_exc()
                            event_queue.put({'type': 'error', 'message': f'Agent error: {str(e)}'})
                        finally:
                            loop.close()
                            event_queue.put(None)  # Always signal completion

                    thread = threading.Thread(target=run_async_stream, daemon=True)
                    thread.start()
                    logger.info("Thread started, beginning event consumption from queue...")

                    # Event types to record for chat history restore
                    _VISUAL_EVENTS = {
                        'planning_complete', 'step_start', 'skill_selection',
                        'clarification_needed', 'prerequisites_needed',
                        'advice', 'warning', 'execution_issue',
                    }
                    visual_events = []

                    # Yield events as they arrive in real-time
                    event_count = 0
                    while True:
                        # Check if user requested abort
                        if chat_abort_event.is_set():
                            logger.info(f"Chat aborted by user after {event_count} events")
                            yield f"data: {json.dumps({'type': 'done', 'message': 'Stopped.'})}\n\n"
                            break

                        try:
                            event_data = event_queue.get(timeout=0.1)
                            if event_data is None:  # Completion signal
                                logger.info(f"Received completion signal (None) after {event_count} events")
                                break
                            event_count += 1
                            event_type = event_data.get('type', 'unknown')
                            if event_type == 'execution_output':
                                logger.debug(f"Yielding SSE event #{event_count}: type={event_type}")
                            else:
                                logger.info(f"Yielding SSE event #{event_count}: type={event_type}")

                            # Record visual events for history restore
                            if event_type in _VISUAL_EVENTS:
                                visual_events.append(event_data)

                            yield f"data: {json.dumps(event_data)}\n\n"
                        except queue.Empty:
                            # No event yet, keep waiting
                            continue

                    # Wait for thread to finish completely (no timeout for long tasks)
                    logger.info("Waiting for thread to complete...")
                    thread.join()
                    logger.info("Thread completed")

                    # Save visual events and continuation flag into turn metadata
                    _turns_after = len(agent.memory.turns) if agent and agent.memory else 0
                    new_turn_created = _turns_after > _turns_before

                    logger.info(f"Turn tracking: before={_turns_before}, after={_turns_after}, new_turn={new_turn_created}, pending_visual={len(_pending_visual_events)}, visual={len(visual_events)}")

                    if new_turn_created and agent is not None:
                        last_turn = agent.memory.turns[-1]
                        if last_turn.metadata is None:
                            last_turn.metadata = {}
                        # Save visual events: separate pending (from clarification request)
                        # and current (from this execution) so frontend can insert reply between them
                        pending = list(_pending_visual_events)
                        _pending_visual_events.clear()
                        if pending or visual_events:
                            last_turn.metadata['visual_events_before'] = pending  # clarification UI
                            last_turn.metadata['visual_events'] = visual_events   # execution events
                            logger.info(f"Saved visual events: {len(pending)} before (clarification), {len(visual_events)} after (execution)")
                        if _is_continuation:
                            last_turn.metadata['is_continuation'] = True
                            if _original_query:
                                last_turn.metadata['original_query'] = _original_query
                            logger.info(f"Marked turn as continuation, original_query={_original_query[:80] if _original_query else None}")
                    elif visual_events:
                        # No turn was created (clarification requested) — save events for next turn
                        _pending_visual_events.extend(visual_events)
                        logger.info(f"No turn created, saved {len(visual_events)} visual events as pending")

                    # Handle celltype color generation if celltypes were updated
                    celltype_updated = len(final_state_changes.get('celltypes_updated', [])) > 0 if final_state_changes else False
                    deconv_updated = len(final_state_changes.get('deconv_weights_updated', [])) > 0 if final_state_changes else False

                    if celltype_updated and session is not None and session.has_data:
                        try:
                            # Generate colors for EACH updated slice (not the stale 'current')
                            updated_slice_ids = final_state_changes.get('celltypes_updated', [])
                            for slice_id in updated_slice_ids:
                                updated_slice = session.get_slice(slice_id)
                                if updated_slice and 'celltype' in updated_slice.adata.obs.columns:
                                    slice_celltypes = sorted(updated_slice.adata.obs['celltype'].unique().tolist())
                                    # Filter out NaN values
                                    slice_celltypes = [ct for ct in slice_celltypes if isinstance(ct, str)]

                                    if slice_celltypes:
                                        # Generate colors for this slice
                                        colors = _generate_color_palette(len(slice_celltypes))
                                        updated_slice.adata.uns['celltype_colors'] = {
                                            ct: colors[i] for i, ct in enumerate(slice_celltypes)
                                        }
                                        logger.info(f"Generated colors for {len(slice_celltypes)} celltypes in slice {slice_id}")
                        except Exception as color_error:
                            logger.error(f"Failed to generate colors: {color_error}")
                            import traceback
                            traceback.print_exc()

                    # Send final completion with state changes
                    # For multi-step plans: plots already shown under each step, don't duplicate
                    # For single-turn: include plots here
                    done_plots = [] if is_multi_step_plan else final_plots
                    logger.info(f"Yielding 'done' event with {len(done_plots)} plots, celltype_updated={celltype_updated}")
                    yield f"data: {json.dumps({'type': 'done', 'plots': done_plots, 'state_changes': final_state_changes or {}, 'celltype_updated': celltype_updated})}\n\n"
                    logger.info("Yielded 'done' event successfully")

                except Exception as e:
                    logger.error(f"Agent error during streaming: {e}")
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Agent error: {str(e)}'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

            else:
                # Fall back to rule-based response
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Using rule-based response...'})}\n\n"
                time.sleep(0.1)

                response = _handle_chat_query(user_message, session)

                yield f"data: {json.dumps({'type': 'response', 'message': response['message'], 'code': response.get('code'), 'agent_powered': False, 'celltype_updated': False})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except GeneratorExit:
            # Client disconnected (page refresh/close) — abort the agent
            logger.info("Client disconnected during chat stream, aborting agent")
            chat_abort_event.set()
            return
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            chat_busy = False

    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response



@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Return conversation history as JSON for frontend reload."""
    global agent

    turns = []
    if agent is not None and hasattr(agent, 'memory'):
        for turn in agent.memory.turns:
            meta = turn.metadata or {}
            turns.append({
                'user': turn.user_message,
                'assistant': turn.assistant_message,
                'code': turn.code_generated,
                'plots': meta.get('plots', []),
                'visual_events_before': meta.get('visual_events_before', []),
                'visual_events': meta.get('visual_events', []),
                'is_continuation': meta.get('is_continuation', False),
                'original_query': meta.get('original_query', None),
                'timestamp': turn.timestamp,
            })

    return jsonify({'success': True, 'turns': turns})


@app.route('/api/session/reset', methods=['POST'])
def reset_session():
    """Reset session and agent state (logout)."""
    global session, agent

    try:
        if agent is not None:
            agent.reset()
        agent = None
        session = None
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to reset session: {e}")
        # Still clear refs even if reset() throws
        agent = None
        session = None
        return jsonify({'success': True})


@app.route('/api/chat/save', methods=['GET'])
def save_chat_history():
    """Generate an HTML file with the complete chat history including plots."""
    global agent

    try:
        # Collect chat history with plots from each turn
        chat_history = []

        if agent is not None and hasattr(agent, 'memory'):
            # Get full conversation turns from agent memory
            for turn in agent.memory.turns:
                # Add user message
                chat_history.append({
                    'type': 'user',
                    'content': turn.user_message,
                    'timestamp': turn.timestamp
                })

                # Add assistant message with its plots
                plots = turn.metadata.get('plots', []) if turn.metadata else []
                chat_history.append({
                    'type': 'assistant',
                    'content': turn.assistant_message,
                    'code': turn.code_generated,
                    'execution_result': turn.execution_result,
                    'plots': plots,  # Include plots from this turn
                    'timestamp': turn.timestamp
                })

        # If no history, return empty
        if not chat_history:
            chat_history = [{
                'type': 'system',
                'content': 'No chat history available.',
                'timestamp': datetime.now().isoformat()
            }]

        # Generate HTML
        html_content = _generate_chat_html(chat_history)

        # Return as downloadable file
        from flask import Response
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stat_agent_chat_{timestamp}.html"

        return Response(
            html_content,
            mimetype='text/html',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )

    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _generate_chat_html(chat_history: List[Dict]) -> str:
    """Generate a self-contained HTML file with chat history and plots."""
    from datetime import datetime

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spatial Agent Chat History</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .header .meta {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .message {{
            background: white;
            margin: 15px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .message-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .message-sender {{
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .message-user .message-sender {{
            color: #667eea;
        }}
        .message-assistant .message-sender {{
            color: #764ba2;
        }}
        .message-system .message-sender {{
            color: #888;
        }}
        .message-timestamp {{
            font-size: 12px;
            color: #999;
        }}
        .message-content {{
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .message-content strong {{
            font-weight: 600;
            color: #333;
        }}
        .code-block {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
        }}
        .execution-result {{
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            padding: 12px 15px;
            margin: 10px 0;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            font-size: 13px;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        .plot-container {{
            margin: 15px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Spatial Agent Chat History</h1>
        <div class="meta">
            Exported: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Total Messages: {len(chat_history)}
        </div>
    </div>
"""

    # Add messages with plots inline
    for msg in chat_history:
        msg_type = msg['type']
        content = msg['content'].replace('\n', '<br>')
        # Bold text: **text** -> <strong>text</strong>
        content = content.replace('**', '<strong>', 1)
        while '<strong>' in content and '</strong>' not in content[content.index('<strong>'):]:
            # Close the bold tag
            next_bold = content.find('**', content.index('<strong>') + 8)
            if next_bold != -1:
                content = content[:next_bold] + '</strong>' + content[next_bold+2:]
            else:
                break

        timestamp = msg.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = dt.strftime("%H:%M:%S")
            except:
                timestamp_str = timestamp[:19] if len(timestamp) >= 19 else timestamp
        else:
            timestamp_str = ''

        sender_label = {
            'user': 'You',
            'assistant': 'Agent',
            'system': 'System'
        }.get(msg_type, msg_type.title())

        html += f"""
    <div class="message message-{msg_type}">
        <div class="message-header">
            <span class="message-sender">{sender_label}</span>
            <span class="message-timestamp">{timestamp_str}</span>
        </div>
        <div class="message-content">{content}</div>
"""

        # Add code block if present
        if msg.get('code'):
            code_escaped = msg['code'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html += f"""
        <div class="code-block">{code_escaped}</div>
"""

        # Add execution result if present
        if msg.get('execution_result'):
            result_escaped = msg['execution_result'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html += f"""
        <div class="execution-result">{result_escaped}</div>
"""

        # Add plots inline with this message (not at the end!)
        plots = msg.get('plots', [])
        if plots:
            for i, plot_base64 in enumerate(plots, 1):
                html += f"""
        <div class="plot-container">
            <img src="data:image/png;base64,{plot_base64}" alt="Plot {i}">
        </div>
"""

        html += "    </div>\n"

    # Footer
    html += """
    <div class="footer">
        Generated by Spatial Transcriptomics Agent<br>
        <a href="https://github.com/yourusername/spatialdata-agent" style="color: #667eea;">GitHub Repository</a>
    </div>
</body>
</html>
"""

    return html


@app.route('/api/test-llm', methods=['POST'])
def test_llm():
    """Test LLM API connection without initializing a full session."""
    if not AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Agent module not available'
        }), 500

    data = request.json
    provider = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')
    base_url = data.get('base_url')

    # Validate inputs
    if not provider:
        return jsonify({
            'success': False,
            'error': 'Provider is required'
        }), 400

    if not api_key or not api_key.strip():
        return jsonify({
            'success': False,
            'error': 'API key is required'
        }), 400

    if not model or not model.strip():
        return jsonify({
            'success': False,
            'error': 'Model name is required'
        }), 400

    try:
        from stat_agent.agent.llm_backend import LLMBackend

        logger.info(f"Testing LLM connection: provider={provider}, model={model}")

        # Build LLM backend kwargs
        llm_kwargs = {
            'system_prompt': 'You are a helpful AI assistant for testing API connections.',
            'model': model.strip(),
            'api_key': api_key.strip()
        }

        # Add base URL if provided
        if base_url and base_url.strip():
            llm_kwargs['endpoint'] = base_url.strip()
            logger.info(f"Using custom endpoint: {base_url.strip()}")

        # Initialize LLM backend
        llm = LLMBackend(**llm_kwargs)

        # Send a simple test query
        test_message = "Hello! Please respond with 'OK' if you can read this message."
        logger.info(f"Sending test message: {test_message}")

        # Run async call in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(
            llm.run(test_message)
        )
        loop.close()

        logger.info(f"Test successful! Response: {response[:100]}...")

        return jsonify({
            'success': True,
            'message': f"Response received: {response[:50]}..." if len(response) > 50 else f"Response: {response}"
        })

    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        import traceback
        traceback.print_exc()

        error_msg = str(e)
        # Provide more helpful error messages
        if 'authentication' in error_msg.lower() or 'api key' in error_msg.lower() or 'unauthorized' in error_msg.lower():
            error_msg = "Authentication failed. Please check your API key."
        elif 'model' in error_msg.lower() and 'not found' in error_msg.lower():
            error_msg = f"Model '{model}' not found. Please check the model name."
        elif 'rate limit' in error_msg.lower():
            error_msg = "Rate limit exceeded. Please wait and try again."
        elif 'timeout' in error_msg.lower():
            error_msg = "Connection timeout. Please check your network connection."

        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


def _handle_chat_query(user_message: str, session: SimpleSession) -> dict:
    """
    Handle user chat query with rule-based responses.

    Can be extended with LLM integration later.
    """
    message_lower = user_message.lower()

    # No session loaded
    if session is None or not session.has_data:
        return {
            'message': "No data loaded yet. Please initialize a session first by providing AnnData and image paths above."
        }

    # Get current slice for data access
    current = session.current_slice()
    if current is None:
        return {
            'message': "No slice selected. Please select a slice first."
        }

    # Data summary queries
    if any(word in message_lower for word in ['what data', 'what is loaded', 'summary', 'info']):
        summary = session.get_frontend_summary()
        msg = f"**Current Session: {summary['name']}**\n\n"
        msg += f"• Cells: {summary['n_cells']:,}\n"
        msg += f"• Genes: {summary['n_genes']:,}\n"

        if summary.get('n_celltypes', 0) > 0:
            msg += f"• Cell types: {summary['n_celltypes']}\n"
            top_celltypes = summary.get('celltypes', [])[:5]
            if top_celltypes:
                msg += f"  - Top types: {', '.join(top_celltypes)}\n"

        if summary.get('n_rois', 0) > 0:
            msg += f"• ROIs: {summary['n_rois']}\n"

        coord_range = summary.get('coordinate_range', {})
        if coord_range:
            msg += f"\n**Spatial extent:**\n"
            msg += f"• X: [{coord_range['x'][0]:.0f}, {coord_range['x'][1]:.0f}]\n"
            msg += f"• Y: [{coord_range['y'][0]:.0f}, {coord_range['y'][1]:.0f}]"

        return {'message': msg}

    # Cell type queries
    if 'cell type' in message_lower or 'celltype' in message_lower:
        if 'celltype' in session.current_slice().adata.obs.columns if session.current_slice() else False:
            celltype_counts = current.adata.obs['celltype'].value_counts()
            msg = f"**Cell Type Distribution** ({current.n_obs:,} total cells):\n\n"
            for ct, count in celltype_counts.head(10).items():
                pct = (count / current.n_obs) * 100
                msg += f"• {ct}: {count:,} ({pct:.1f}%)\n"

            if len(celltype_counts) > 10:
                msg += f"\n... and {len(celltype_counts) - 10} more types"

            return {'message': msg}
        else:
            return {'message': "Cell type information is not available in this dataset."}

    # ROI queries
    if 'roi' in message_lower:
        if len(session.roi_manager.rois) > 0:
            rois = list(session.roi_manager.rois.values())
            msg = f"**Regions of Interest** ({len(rois)} ROIs):\n\n"
            for roi in rois:
                msg += f"• {roi.name} ({roi.type})\n"
                if roi.bounds:
                    b = roi.bounds
                    msg += f"  Bounds: ({b[0]:.0f}, {b[1]:.0f}) to ({b[2]:.0f}, {b[3]:.0f})\n"
                if roi.name in session.roi_subsets:
                    n_cells = session.roi_subsets[roi.name].n_obs
                    msg += f"  Cells: {n_cells:,}\n"
            return {'message': msg}
        else:
            return {'message': "No ROIs created yet. Draw an ROI on the canvas to analyze a specific region."}

    # Analysis suggestions
    if 'suggest' in message_lower or 'help' in message_lower or 'what can' in message_lower:
        msg = "**Analysis Suggestions:**\n\n"
        msg += "📊 **Explore your data:**\n"
        msg += "• Ask 'What data is loaded?' to see a summary\n"
        msg += "• Ask 'Show cell types' to see cell type distribution\n\n"

        msg += "🎯 **Region selection:**\n"
        msg += "• Click 'Start Drawing' to define a Region of Interest (ROI)\n"
        msg += "• Click two points to define a bounding box\n"
        msg += "• Click 'Finish' to select cells in that region\n\n"

        msg += "🔬 **Cell overlay:**\n"
        msg += "• Click 'Load Cell Overlay' to visualize cell positions\n"
        msg += "• Click 'Clear Cells' to remove the overlay\n\n"

        msg += "💡 **Tips:**\n"
        msg += "• Use mouse wheel to zoom in/out\n"
        msg += "• Drag to pan the image\n"
        msg += "• ROIs show cell counts and cell type distributions"

        return {'message': msg}

    # Gene queries
    if 'gene' in message_lower:
        if current.adata.var_names is not None:
            n_genes = current.n_vars
            msg = f"**Gene Information:**\n\n"
            msg += f"• Total genes: {n_genes:,}\n\n"
            msg += "Sample genes:\n"
            for gene in current.adata.var_names[:10]:
                msg += f"• {gene}\n"
            if n_genes > 10:
                msg += f"\n... and {n_genes - 10:,} more genes"
            return {'message': msg}
        else:
            return {'message': "Gene information is not available in this dataset."}

    # Default response
    return {
        'message': f"I understand you're asking: '{user_message}'\n\n"
                   "I can help with:\n"
                   "• Data summaries ('What data is loaded?')\n"
                   "• Cell type analysis ('Show cell types')\n"
                   "• ROI information ('List ROIs')\n"
                   "• Analysis suggestions ('Suggest an analysis')\n\n"
                   "Try asking one of these questions!"
    }


def main():
    """Run the web interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Simplified Spatial Agent Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--jupyter-port', type=int, default=8890, help='Jupyter Lab port')

    args = parser.parse_args()

    app.config['JUPYTER_PORT'] = args.jupyter_port

    logger.info(f"Starting Simplified Spatial Agent Web Interface on {args.host}:{args.port}")
    logger.info(f"Access the interface at: http://localhost:{args.port}")
    logger.info(f"Jupyter Lab expected on port: {args.jupyter_port}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
