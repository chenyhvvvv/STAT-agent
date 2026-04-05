(function() {
  'use strict';

  // ============================================================
  // SECTION 1: STATE & CONSTANTS
  // ============================================================

  const state = {
    initialized: false,
    sessionSummary: null,
    currentSliceId: null,
    currentModality: 'gene',
    geneList: [],

    // ROI
    roiTool: null,             // null | 'bbox' | 'polygon' | 'freehand'
    roiDrawingState: null,     // in-progress drawing coords
    rois: [],
    highlightedROI: null,      // name of ROI being hovered in list

    // Chat
    chatMessages: [],
    chatStreaming: false,
    chatAbortController: null,
    currentAssistantMsg: null,  // DOM element being built during streaming

    // Canvas cache: keyed by "slice_{id}_{modality}"
    canvases: {},

    // Interaction
    isPanning: false,
    panStart: null,
  };

  // ---- localStorage helpers for form persistence ----
  const STORAGE_PREFIX = 'stat_';
  const PERSISTED_FIELDS = ['dataset-dir', 'session-name', 'api-key', 'model', 'base-url'];

  function saveFormToStorage() {
    PERSISTED_FIELDS.forEach(id => {
      const el = document.getElementById(id);
      if (el) localStorage.setItem(STORAGE_PREFIX + id, el.value);
    });
  }

  function loadFormFromStorage() {
    PERSISTED_FIELDS.forEach(id => {
      const el = document.getElementById(id);
      const saved = localStorage.getItem(STORAGE_PREFIX + id);
      if (el && saved) el.value = saved;
    });
  }

  // ---- Session reconnect on page refresh ----
  async function tryReconnectSession() {
    try {
      const res = await api.getSessionSummary();
      if (res.success && res.summary) {
        // Backend session still alive — skip init form
        state.sessionSummary = res.summary;
        state.currentSliceId = res.summary.current_slice_id;
        state.currentModality = res.summary.current_modality || 'gene';
        state.initialized = true;

        document.getElementById('init-overlay').classList.add('hidden');
        document.getElementById('app').classList.remove('hidden');

        const agentActive = res.agent_active !== false;
        document.getElementById('send-btn').disabled = !agentActive;
        document.getElementById('chat-input').placeholder = agentActive
          ? 'Ask about your data...'
          : 'Agent not available (no API key)';

        updateSessionInfo();
        updateSliceTabs();
        updateModalityToggle();
        updateVizModeState();
        setupResizeObserver();
        await loadSliceData(state.currentSliceId, state.currentModality);
        updateCelltypeCheckboxes();
        updateDataInfo();
        updateVizModeState();

        api.getGeneList().then(r => {
          if (r.success) state.geneList = r.genes || [];
        });
        refreshROIs();
        restoreChatHistory();

        // If agent was busy when we reconnected, show a notice
        if (res.chat_busy) {
          const chatEl = document.getElementById('chat-messages');
          const notice = document.createElement('div');
          notice.className = 'chat-msg system';
          notice.innerHTML = '<div class="chat-msg-body small-info">A previous request was interrupted. You can send a new message.</div>';
          chatEl.appendChild(notice);
          scrollChatToBottom();
        }

        setTimeout(preloadAllSlices, 500);
        return true;
      }
    } catch (_) { /* backend not ready or no session */ }
    return false;
  }

  // --- History restore helpers ---

  function _renderVisualEvents(body, events) {
    if (!events || !events.length) return;
    for (const ev of events) {
      switch (ev.type) {
        case 'planning_complete': {
          const steps = Array.isArray(ev.plan || ev.steps) ? (ev.plan || ev.steps) : ((ev.plan && ev.plan.steps) ? ev.plan.steps : []);
          if (steps.length) {
            const stepsHtml = steps.map((s, i) => {
              const num = s.step_number || (i + 1);
              return `<li class="plan-step done" data-step="${num}"><span class="plan-step-num">${num}</span>${escapeHtml(s.description || '')}</li>`;
            }).join('');
            body.insertAdjacentHTML('beforeend', `<details class="plan-card"><summary>Plan (${steps.length} steps)</summary><ol class="plan-steps">${stepsHtml}</ol></details>`);
          }
          break;
        }
        case 'step_start':
          body.insertAdjacentHTML('beforeend', `<div class="small-info" style="margin:6px 0;font-weight:600">Step ${ev.step_number}${ev.total_steps ? '/' + ev.total_steps : ''}: ${escapeHtml(ev.description || '')}</div>`);
          break;
        case 'skill_selection': {
          const options = ev.options || [];
          const msg = ev.message || 'Multiple skills matched.';
          const optHtml = options.map((o, i) =>
            `<button class="clarification-option-btn" disabled>${i + 1}. ${escapeHtml(o.name || o.slug)}</button>`
          ).join('');
          body.insertAdjacentHTML('beforeend', `<div class="clarification-ui"><div class="question">${escapeHtml(msg)}</div><div class="clarification-options">${optHtml}</div></div>`);
          break;
        }
        case 'clarification_needed': {
          let html = `<div class="clarification-ui"><div class="question">${renderMarkdown(ev.question || '')}</div>`;
          if (ev.options && ev.options.length) {
            html += '<div class="clarification-options">';
            ev.options.forEach(o => {
              const label = typeof o === 'string' ? o : (o.label || o.name || o);
              html += `<button class="clarification-option-btn" disabled>${escapeHtml(label)}</button>`;
            });
            html += '</div>';
          }
          html += '</div>';
          body.insertAdjacentHTML('beforeend', html);
          break;
        }
        case 'prerequisites_needed': {
          let html = `<div class="clarification-ui"><div class="question">Prerequisites needed for <strong>${escapeHtml(ev.skill || 'skill')}</strong>:</div>`;
          html += '<div class="clarification-options">';
          (ev.questions || []).forEach(q => {
            html += `<div style="margin:4px 0"><label style="font-size:12px">${escapeHtml(q)}</label></div>`;
          });
          html += '</div></div>';
          body.insertAdjacentHTML('beforeend', html);
          break;
        }
        case 'advice':
          body.insertAdjacentHTML('beforeend', `<div class="alert-card advice">${renderMarkdown(ev.message)}</div>`);
          break;
        case 'warning':
          body.insertAdjacentHTML('beforeend', `<div class="alert-card warning">${renderMarkdown(ev.message)}</div>`);
          break;
        case 'execution_issue':
          body.insertAdjacentHTML('beforeend', `<div class="alert-card error"><strong>Execution Issue</strong> (${escapeHtml(ev.issue_type || 'error')})<br>${renderMarkdown(ev.explanation || '')}</div>`);
          break;
      }
    }
  }

  function _renderAssistantContent(body, turn) {
    // Render turn.assistant markdown into body, matching live streaming output:
    // - Skip python code fences (not shown during streaming)
    // - Output fences → collapsible <details>
    // - Plot markers → inline plots at correct position
    // - **Error:** → alert card
    // - **Analysis:** → styled header (skipped if had error)
    if (!turn.assistant) return;

    const plots = turn.plots || [];
    let plotIdx = 0;
    let hasError = (turn.visual_events || []).some(e => e.type === 'execution_issue');

    const parts = turn.assistant.split(/(```python\n[\s\S]*?```|```\n[\s\S]*?```)/g);
    for (const part of parts) {
      if (/^```python\n/.test(part)) continue;

      const outputMatch = part.match(/^```\n([\s\S]*?)```$/);
      if (outputMatch) {
        const details = document.createElement('details');
        details.className = 'exec-details';
        const summary = document.createElement('summary');
        summary.textContent = 'Output';
        details.appendChild(summary);
        const pre = document.createElement('div');
        pre.className = 'exec-output';
        pre.textContent = outputMatch[1];
        details.appendChild(pre);
        body.appendChild(details);
        continue;
      }

      // Text — split off **Analysis:** (always at end), then process lines
      let textBody = part;
      let analysisBlock = null;
      const analysisSplit = textBody.split(/\n(\*\*Analysis:?\*\*\n[\s\S]*)$/);
      if (analysisSplit.length > 1) {
        textBody = analysisSplit[0];
        analysisBlock = analysisSplit[1];
      }

      const lines = textBody.split('\n');
      let textBuf = '';
      for (const line of lines) {
        const plotMatch = line.match(/^\*?\((\d+) plot\(s\) generated\)\*?$/);
        if (plotMatch) {
          if (textBuf.trim()) { const d = document.createElement('div'); d.innerHTML = renderMarkdown(textBuf.trim()); body.appendChild(d); textBuf = ''; }
          const nPlots = parseInt(plotMatch[1], 10) || 0;
          for (let p = 0; p < nPlots && plotIdx < plots.length; p++, plotIdx++) {
            const src = plots[plotIdx];
            const imgSrc = (typeof src === 'string' && src.startsWith('data:')) ? src : `data:image/png;base64,${src}`;
            const d = document.createElement('div'); d.className = 'chat-plot'; d.innerHTML = `<img src="${imgSrc}" alt="Plot">`; body.appendChild(d);
          }
          continue;
        }
        const errorMatch = line.match(/^\*\*Error:?\*\*\s*(.*)/);
        if (errorMatch) {
          if (textBuf.trim()) { const d = document.createElement('div'); d.innerHTML = renderMarkdown(textBuf.trim()); body.appendChild(d); textBuf = ''; }
          hasError = true;
          const d = document.createElement('div'); d.className = 'alert-card error';
          d.innerHTML = '<strong>Execution Error</strong><br>' + renderMarkdown(errorMatch[1]); body.appendChild(d);
          continue;
        }
        textBuf += line + '\n';
      }
      if (textBuf.trim()) { const d = document.createElement('div'); d.innerHTML = renderMarkdown(textBuf.trim()); body.appendChild(d); }

      if (analysisBlock && !hasError) {
        const aMatch = analysisBlock.match(/^\*\*Analysis:?\*\*\s*\n?([\s\S]*)$/);
        if (aMatch) {
          const ab = aMatch[1].trim();
          if (ab) {
            const d = document.createElement('div');
            d.innerHTML = '<hr style="border:none;border-top:1px solid var(--border-light);margin:10px 0"><div><em><strong>Analysis</strong></em></div><div>' + renderMarkdown(ab) + '</div>';
            body.appendChild(d);
          }
        }
      }
    }

    // Remaining plots
    while (plotIdx < plots.length) {
      const src = plots[plotIdx++];
      const imgSrc = (typeof src === 'string' && src.startsWith('data:')) ? src : `data:image/png;base64,${src}`;
      const d = document.createElement('div'); d.className = 'chat-plot'; d.innerHTML = `<img src="${imgSrc}" alt="Plot">`; body.appendChild(d);
    }
  }

  async function restoreChatHistory() {
    try {
      const res = await api.getChatHistory();
      if (!res.success || !res.turns || res.turns.length === 0) return;

      const chatEl = document.getElementById('chat-messages');
      const welcome = chatEl.querySelector('.chat-welcome');
      if (welcome) welcome.remove();

      for (const turn of res.turns) {
        // For continuation turns (clarification/skill selection/prerequisite replies):
        // In the live UI, the original query → clarification → user reply → result all
        // happened in ONE user bubble + ONE assistant bubble. No separate turn was created
        // for the clarification question — it was just a visual event.
        //
        // In memory, only the resolution turn exists: Turn(user="1", assistant="results").
        // The original query is saved in turn.original_query.
        //
        // We reconstruct the full exchange: original query as user bubble, then one
        // assistant bubble containing: [clarification UI] → [inline reply] → [results].

        // User bubble: show original_query for continuations, otherwise turn.user
        const userText = (turn.is_continuation && turn.original_query) ? turn.original_query : turn.user;
        const userDiv = document.createElement('div');
        userDiv.className = 'chat-msg user';
        userDiv.innerHTML = `<div class="chat-msg-body">${escapeHtml(userText)}</div>`;
        chatEl.appendChild(userDiv);

        // Assistant bubble
        const assistDiv = document.createElement('div');
        assistDiv.className = 'chat-msg assistant';
        const body = document.createElement('div');
        body.className = 'chat-msg-body';

        if (turn.is_continuation) {
          // Render: [clarification UI] → [user's reply] → [execution events] → [results]
          // visual_events_before = from the clarification request (skill selection, clarification, etc.)
          // visual_events = from this execution (planning, steps, etc.)
          _renderVisualEvents(body, turn.visual_events_before);
          body.insertAdjacentHTML('beforeend',
            `<div class="small-info" style="margin:6px 0;color:var(--text-secondary);font-style:italic">↳ ${escapeHtml(turn.user)}</div>`);
          _renderVisualEvents(body, turn.visual_events);
        } else {
          _renderVisualEvents(body, turn.visual_events_before);
          _renderVisualEvents(body, turn.visual_events);
        }

        _renderAssistantContent(body, turn);

        assistDiv.appendChild(body);
        chatEl.appendChild(assistDiv);
      }
      scrollChatToBottom();
    } catch (_) { /* no history available */ }
  }

  async function logout() {
    // Call backend reset to clear memory/session
    try { await api.resetSession(); } catch (_) {}

    // Reset state
    state.initialized = false;
    state.sessionSummary = null;
    state.currentSliceId = null;
    state.currentModality = 'gene';
    state.geneList = [];
    state.rois = [];
    state.chatMessages = [];
    state.chatStreaming = false;
    state.canvases = {};
    state.currentAssistantMsg = null;

    // Show init, hide app
    document.getElementById('app').classList.add('hidden');
    document.getElementById('init-overlay').classList.remove('hidden');

    // Clear chat messages
    const chatEl = document.getElementById('chat-messages');
    chatEl.innerHTML = '<div class="chat-welcome"><p>Ask questions about your spatial transcriptomics data.</p><p class="small-info">Examples: "Annotate cell types", "Find spatially variable genes", "Show BRCA1 expression"</p></div>';

    // Restore saved form values
    loadFormFromStorage();
  }

  const CELL_RADIUS = 3;
  const SPOT_SCALE = 0.5;       // spot_diameter * SPOT_SCALE = canvas radius
  const MIN_ZOOM = 0.1;
  const MAX_ZOOM = 50;
  const ZOOM_FACTOR = 1.15;

  // Viridis-like color LUT (256 entries)
  let VIRIDIS_LUT = null;

  function buildViridisLUT() {
    // Simplified viridis: dark purple -> teal -> yellow
    const stops = [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37]
    ];
    const lut = new Array(256);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      const seg = t * (stops.length - 1);
      const idx = Math.min(Math.floor(seg), stops.length - 2);
      const frac = seg - idx;
      const a = stops[idx], b = stops[idx + 1];
      lut[i] = `rgb(${Math.round(a[0] + (b[0]-a[0])*frac)},${Math.round(a[1] + (b[1]-a[1])*frac)},${Math.round(a[2] + (b[2]-a[2])*frac)})`;
    }
    VIRIDIS_LUT = lut;
  }

  // ============================================================
  // SECTION 2: UTILITY FUNCTIONS
  // ============================================================

  function getCanvasKey(sliceId, modality) {
    return `slice_${sliceId}_${modality}`;
  }

  function getCurrentCanvas() {
    const key = getCanvasKey(state.currentSliceId, state.currentModality);
    if (!state.canvases[key]) {
      state.canvases[key] = createEmptyCanvas();
    }
    return state.canvases[key];
  }

  function createEmptyCanvas() {
    return {
      image: null, imageWidth: 0, imageHeight: 0, isScatter: false,
      viewX: 0, viewY: 0, viewWidth: 0, viewHeight: 0, zoom: 1.0,
      cells: null, nCells: 0, hasCelltype: false,
      celltypes: [], celltypeColors: {}, selectedCelltypes: new Set(),
      geneExpression: null, geneExpressionRange: null,
      customVmax: null,  // user-controlled max for color scale (null = auto)
      isSpotData: false, spotInfo: null,
      // Per-slice view settings
      vizMode: 'celltype',    // 'celltype' | 'gene' | 'proportion'
      selectedGene: null,
      opacity: 0.8,           // cell overlay opacity (0-1)
      pointSize: 1.0,         // point size multiplier (0.2-3)
      hideBackground: false,  // hide tissue image
      proportionCelltype: null, // single celltype for proportion heatmap (null = pie chart)
      loaded: false, loading: false,
    };
  }

  function getCurrentROIs() {
    return state.rois.filter(r =>
      r.slice_id === state.currentSliceId &&
      r.modality === state.currentModality
    );
  }

  // Per-canvas accessors for settings that used to be global
  function getVizMode() { const c = getCurrentCanvas(); return c ? c.vizMode : 'celltype'; }
  function getSelectedGene() { const c = getCurrentCanvas(); return c ? c.selectedGene : null; }

  function imageToCanvas(imgX, imgY) {
    const c = getCurrentCanvas();
    const el = document.getElementById('spatial-canvas');
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    return { x: (imgX - c.viewX) * sx, y: (imgY - c.viewY) * sy };
  }

  function canvasToImage(cx, cy) {
    const c = getCurrentCanvas();
    const el = document.getElementById('spatial-canvas');
    const sx = c.viewWidth / el.width;
    const sy = c.viewHeight / el.height;
    return { x: c.viewX + cx * sx, y: c.viewY + cy * sy };
  }

  function debounce(fn, ms) {
    let timer;
    return function(...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // Global color registry — ensures same celltype name = same color across all slices
  const _globalCelltypeColors = {};

  function getOrAssignColor(name, canvasColors) {
    // 1. Already in global registry
    if (_globalCelltypeColors[name]) return _globalCelltypeColors[name];
    // 2. Backend assigned a color on some canvas — adopt it globally
    if (canvasColors && canvasColors[name]) {
      _globalCelltypeColors[name] = canvasColors[name];
      return canvasColors[name];
    }
    // 3. Check all loaded canvases for an existing color
    for (const key of Object.keys(state.canvases)) {
      const cc = state.canvases[key].celltypeColors;
      if (cc && cc[name]) {
        _globalCelltypeColors[name] = cc[name];
        return cc[name];
      }
    }
    // 4. Generate a stable color from name hash
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
      hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
    }
    const hue = ((Math.abs(hash) * 0.618033988749895) % 1) * 360;
    const s = 65 + (Math.abs(hash >> 8) % 20);
    const l = 45 + (Math.abs(hash >> 16) % 15);
    const color = `hsl(${Math.round(hue)}, ${s}%, ${l}%)`;
    _globalCelltypeColors[name] = color;
    return color;
  }

  // Kept for backward compat — delegates to global registry
  function generateColor(name) {
    return getOrAssignColor(name, null);
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function expressionToColor(value, min, max) {
    if (!VIRIDIS_LUT) buildViridisLUT();
    if (max === min) return VIRIDIS_LUT[0];
    const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return VIRIDIS_LUT[Math.round(t * 255)];
  }

  function showSpinner(msg) {
    const el = document.getElementById('canvas-spinner');
    el.querySelector('span').textContent = msg || 'Loading...';
    el.classList.remove('hidden');
  }

  function hideSpinner() {
    document.getElementById('canvas-spinner').classList.add('hidden');
  }

  // ============================================================
  // SECTION 3: API CLIENT
  // ============================================================

  const api = {
    async initDataset(params) {
      const res = await fetch('/api/init_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      return res.json();
    },

    async getSessionSummary() {
      const res = await fetch('/api/session/summary');
      return res.json();
    },

    async getChatHistory() {
      const res = await fetch('/api/chat/history');
      return res.json();
    },

    async resetSession() {
      const res = await fetch('/api/session/reset', { method: 'POST' });
      return res.json();
    },

    async getImageData(sliceId, modality) {
      const params = new URLSearchParams();
      if (sliceId != null) params.set('slice_id', sliceId);
      if (modality) params.set('modality', modality);
      const res = await fetch(`/api/image/data?${params}`);
      return res.json();
    },

    async getCellOverlay(sliceId, modality, selectedCelltypes) {
      const params = new URLSearchParams();
      if (sliceId != null) params.set('slice_id', sliceId);
      if (modality) params.set('modality', modality);
      if (selectedCelltypes) params.set('selected_celltypes', selectedCelltypes);
      const res = await fetch(`/api/cells/overlay?${params}`);
      return res.json();
    },

    async getGeneExpression(gene) {
      const params = new URLSearchParams({ gene });
      const res = await fetch(`/api/cells/gene_expression?${params}`);
      return res.json();
    },

    async getGeneList() {
      const res = await fetch('/api/genes/list');
      return res.json();
    },

    async selectSlice(sliceId) {
      const res = await fetch('/api/select_slice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slice_id: sliceId }),
      });
      return res.json();
    },

    async selectModality(modality) {
      const res = await fetch('/api/select_modality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modality }),
      });
      return res.json();
    },

    async addROI(name, type, params) {
      const res = await fetch('/api/roi/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, type, params }),
      });
      return res.json();
    },

    async listROIs() {
      const res = await fetch('/api/roi/list');
      return res.json();
    },

    async testLLM(config) {
      const res = await fetch('/api/test-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      return res.json();
    },

    async deleteROI(name) {
      const res = await fetch('/api/roi/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      return res.json();
    },

    async renameROI(oldName, newName) {
      const res = await fetch('/api/roi/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      });
      return res.json();
    },

    async chatAbort() {
      await fetch('/api/chat/abort', { method: 'POST' });
    },

    async getNotebookURL() {
      const res = await fetch('/api/notebook/url');
      return res.json();
    },

    saveChatHistory() {
      window.open('/api/chat/save', '_blank');
    },
  };

  // ============================================================
  // SECTION 4: CANVAS RENDERING ENGINE
  // ============================================================

  function render() {
    const t0 = performance.now();
    const el = document.getElementById('spatial-canvas');
    if (!el) return;
    const ctx = el.getContext('2d');
    const c = getCurrentCanvas();
    if (!c || !c.loaded) {
      ctx.clearRect(0, 0, el.width, el.height);
      return;
    }

    ctx.clearRect(0, 0, el.width, el.height);
    ctx.save();

    // Draw image (or white background)
    if (c.image && !c.hideBackground) {
      ctx.drawImage(
        c.image,
        c.viewX, c.viewY, c.viewWidth, c.viewHeight,
        0, 0, el.width, el.height
      );
    } else {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, el.width, el.height);
    }

    // Draw cells/spots with per-canvas opacity
    ctx.globalAlpha = c.opacity;
    if (c.vizMode === 'celltype') {
      renderCells(ctx, c, el);
    } else if (c.vizMode === 'gene' && c.geneExpression) {
      renderGeneExpression(ctx, c, el);
    } else if (c.vizMode === 'proportion' && c.isSpotData && c.spotInfo) {
      renderProportions(ctx, c, el);
    }
    ctx.globalAlpha = 1.0;

    // Draw ROIs
    renderROIs(ctx, el);

    // Draw in-progress ROI
    if (state.roiDrawingState) {
      renderROIDrawing(ctx, el);
    }

    ctx.restore();

    // Update legend
    updateLegend(c);
    const dt = performance.now() - t0;
    if (dt > 10) console.log(`render: ${dt.toFixed(1)}ms`);
  }

  function renderCells(ctx, c, el) {
    if (!c.cells || !c.cells.x) return;
    const xs = c.cells.x, ys = c.cells.y, cts = c.cells.celltype;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.isSpotData && c.spotInfo ? (c.spotInfo.spot_diameter * SPOT_SCALE) : CELL_RADIUS) * c.pointSize;
    const r = Math.max(rImg * sx, 1);
    const useCircles = r > 3; // Only draw circles when zoomed in enough to see them
    const cullPad = rImg + 1;

    // Batch by color for performance
    const batches = {};
    const vxMin = c.viewX - cullPad, vxMax = c.viewX + c.viewWidth + cullPad;
    const vyMin = c.viewY - cullPad, vyMax = c.viewY + c.viewHeight + cullPad;

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < vxMin || x > vxMax || y < vyMin || y > vyMax) continue;

      const ct = cts ? cts[i] : null;
      if (ct && c.selectedCelltypes.size > 0 && !c.selectedCelltypes.has(ct)) continue;

      const color = (ct && c.celltypeColors[ct]) || 'rgb(150,150,150)';
      if (!batches[color]) batches[color] = [];
      batches[color].push((x - c.viewX) * sx, (y - c.viewY) * sy);
    }

    if (useCircles) {
      // Zoomed in: draw circles (fewer visible cells due to frustum cull)
      for (const [color, coords] of Object.entries(batches)) {
        ctx.beginPath();
        for (let i = 0; i < coords.length; i += 2) {
          ctx.moveTo(coords[i] + r, coords[i+1]);
          ctx.arc(coords[i], coords[i+1], r, 0, Math.PI * 2);
        }
        ctx.fillStyle = color;
        ctx.fill();
      }
    } else {
      // Zoomed out: draw fast rectangles (310k rects is ~50x faster than 310k arcs)
      const size = Math.max(Math.round(r * 2), 1);
      for (const [color, coords] of Object.entries(batches)) {
        ctx.fillStyle = color;
        for (let i = 0; i < coords.length; i += 2) {
          ctx.fillRect(coords[i] - r, coords[i+1] - r, size, size);
        }
      }
    }
  }

  function renderGeneExpression(ctx, c, el) {
    const ge = c.geneExpression;
    if (!ge || !ge.x) return;
    const xs = ge.x, ys = ge.y, expr = ge.expression;
    const [emin, rawMax] = c.geneExpressionRange || [0, 1];
    const emax = c.customVmax != null ? c.customVmax : rawMax;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.isSpotData && c.spotInfo ? (c.spotInfo.spot_diameter * SPOT_SCALE) : CELL_RADIUS) * c.pointSize;
    const r = Math.max(rImg * sx, 1);
    const useCircles = r > 3;
    const cullPad = rImg + 1;
    const vxMin = c.viewX - cullPad, vxMax = c.viewX + c.viewWidth + cullPad;
    const vyMin = c.viewY - cullPad, vyMax = c.viewY + c.viewHeight + cullPad;

    // Bucket by color index for batching
    if (!VIRIDIS_LUT) buildViridisLUT();
    const buckets = new Array(256);
    for (let b = 0; b < 256; b++) buckets[b] = [];

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < vxMin || x > vxMax || y < vyMin || y > vyMax) continue;

      const t = emax === emin ? 0 : (expr[i] - emin) / (emax - emin);
      const bi = Math.max(0, Math.min(255, Math.round(t * 255)));
      buckets[bi].push((x - c.viewX) * sx, (y - c.viewY) * sy);
    }

    const size = Math.max(Math.round(r * 2), 1);
    for (let b = 0; b < 256; b++) {
      const coords = buckets[b];
      if (coords.length === 0) continue;
      ctx.fillStyle = VIRIDIS_LUT[b];
      if (useCircles) {
        ctx.beginPath();
        for (let i = 0; i < coords.length; i += 2) {
          ctx.moveTo(coords[i] + r, coords[i+1]);
          ctx.arc(coords[i], coords[i+1], r, 0, Math.PI * 2);
        }
        ctx.fill();
      } else {
        for (let i = 0; i < coords.length; i += 2) {
          ctx.fillRect(coords[i] - r, coords[i+1] - r, size, size);
        }
      }
    }
  }

  function renderProportions(ctx, c, el) {
    if (!c.cells || !c.spotInfo || !c.spotInfo.deconv_weights) return;

    // Single-celltype heatmap mode
    if (c.proportionCelltype) {
      renderProportionHeatmap(ctx, c, el);
      return;
    }

    // Pie chart mode
    const xs = c.cells.x, ys = c.cells.y;
    const weights = c.spotInfo.deconv_weights;
    const order = c.spotInfo.celltype_order || Object.keys(weights);
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.spotInfo.spot_diameter ? (c.spotInfo.spot_diameter * SPOT_SCALE) : 8) * c.pointSize;
    const r = Math.max(rImg * sx, 2);

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < c.viewX || x > c.viewX + c.viewWidth ||
          y < c.viewY || y > c.viewY + c.viewHeight) continue;

      const cx = (x - c.viewX) * sx;
      const cy = (y - c.viewY) * sy;

      let total = 0;
      for (const ct of order) {
        total += (weights[ct] ? (weights[ct][i] || 0) : 0);
      }
      if (total === 0) continue;

      let startAngle = -Math.PI / 2;
      for (const ct of order) {
        const val = weights[ct] ? (weights[ct][i] || 0) : 0;
        const prop = val / total;
        if (prop < 0.01) { startAngle += prop * Math.PI * 2; continue; }
        const sliceAngle = prop * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, startAngle, startAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = c.celltypeColors[ct] || '#999';
        ctx.fill();
        startAngle += sliceAngle;
      }
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,0,0,0.2)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  function renderProportionHeatmap(ctx, c, el) {
    const ct = c.proportionCelltype;
    const weights = c.spotInfo.deconv_weights;
    if (!weights[ct]) return;
    const xs = c.cells.x, ys = c.cells.y;
    const vals = weights[ct];
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;
    const rImg = (c.isSpotData && c.spotInfo ? (c.spotInfo.spot_diameter * SPOT_SCALE) : CELL_RADIUS) * c.pointSize;
    const r = Math.max(rImg * sx, 1);
    const useCircles = r > 3;
    const vmax = c.customVmax != null ? c.customVmax : 1.0;
    const cullPad = rImg + 1;
    const vxMin = c.viewX - cullPad, vxMax = c.viewX + c.viewWidth + cullPad;
    const vyMin = c.viewY - cullPad, vyMax = c.viewY + c.viewHeight + cullPad;

    if (!VIRIDIS_LUT) buildViridisLUT();
    const buckets = new Array(256);
    for (let b = 0; b < 256; b++) buckets[b] = [];

    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x < vxMin || x > vxMax || y < vyMin || y > vyMax) continue;
      const val = vals[i] || 0;
      const t = vmax === 0 ? 0 : Math.min(1, val / vmax);
      const bi = Math.max(0, Math.min(255, Math.round(t * 255)));
      buckets[bi].push((x - c.viewX) * sx, (y - c.viewY) * sy);
    }

    const size = Math.max(Math.round(r * 2), 1);
    for (let b = 0; b < 256; b++) {
      const coords = buckets[b];
      if (coords.length === 0) continue;
      ctx.fillStyle = VIRIDIS_LUT[b];
      if (useCircles) {
        ctx.beginPath();
        for (let i = 0; i < coords.length; i += 2) {
          ctx.moveTo(coords[i] + r, coords[i+1]);
          ctx.arc(coords[i], coords[i+1], r, 0, Math.PI * 2);
        }
        ctx.fill();
      } else {
        for (let i = 0; i < coords.length; i += 2) {
          ctx.fillRect(coords[i] - r, coords[i+1] - r, size, size);
        }
      }
    }
  }

  function renderROIs(ctx, el) {
    const rois = getCurrentROIs();
    const c = getCurrentCanvas();
    if (!c || rois.length === 0) return;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;

    for (const roi of rois) {
      const isHighlighted = state.highlightedROI === roi.name;
      ctx.save();

      // Highlighted ROI: solid thick line, stronger fill
      if (isHighlighted) {
        ctx.setLineDash([]);
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.9)';
        ctx.fillStyle = 'rgba(220, 38, 38, 0.12)';
      } else {
        ctx.setLineDash([6, 3]);
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.fillStyle = 'rgba(245, 158, 11, 0.1)';
      }

      if (roi.type === 'bbox' && roi.bounds) {
        const [x1, y1, x2, y2] = roi.bounds;
        const rx = (x1 - c.viewX) * sx;
        const ry = (y1 - c.viewY) * sy;
        const rw = (x2 - x1) * sx;
        const rh = (y2 - y1) * sy;
        ctx.fillRect(rx, ry, rw, rh);
        ctx.strokeRect(rx, ry, rw, rh);
      } else if (roi.vertices && roi.vertices.length > 0) {
        ctx.beginPath();
        const [fx, fy] = roi.vertices[0];
        ctx.moveTo((fx - c.viewX) * sx, (fy - c.viewY) * sy);
        for (let i = 1; i < roi.vertices.length; i++) {
          const [vx, vy] = roi.vertices[i];
          ctx.lineTo((vx - c.viewX) * sx, (vy - c.viewY) * sy);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Label — with background for readability
      if (roi.name) {
        const bx = roi.bounds ? roi.bounds[0] : (roi.vertices ? roi.vertices[0][0] : 0);
        const by = roi.bounds ? roi.bounds[1] : (roi.vertices ? roi.vertices[0][1] : 0);
        const lx = (bx - c.viewX) * sx + 4;
        const ly = (by - c.viewY) * sy - 6;
        ctx.setLineDash([]);
        ctx.font = 'bold 12px sans-serif';
        const textW = ctx.measureText(roi.name).width;
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fillRect(lx - 2, ly - 11, textW + 4, 14);
        ctx.fillStyle = isHighlighted ? 'rgba(220, 38, 38, 1)' : 'rgba(180, 100, 0, 1)';
        ctx.fillText(roi.name, lx, ly);
      }

      ctx.restore();
    }
  }

  function renderROIDrawing(ctx, el) {
    const ds = state.roiDrawingState;
    const c = getCurrentCanvas();
    if (!ds || !c) return;
    const sx = el.width / c.viewWidth;
    const sy = el.height / c.viewHeight;

    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(220, 38, 38, 0.9)';
    ctx.fillStyle = 'rgba(220, 38, 38, 0.08)';

    if (ds.type === 'bbox' && ds.start && ds.current) {
      const x1 = Math.min(ds.start.x, ds.current.x);
      const y1 = Math.min(ds.start.y, ds.current.y);
      const x2 = Math.max(ds.start.x, ds.current.x);
      const y2 = Math.max(ds.start.y, ds.current.y);
      const rx = (x1 - c.viewX) * sx;
      const ry = (y1 - c.viewY) * sy;
      const rw = (x2 - x1) * sx;
      const rh = (y2 - y1) * sy;
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeRect(rx, ry, rw, rh);
    } else if ((ds.type === 'polygon' || ds.type === 'freehand') && ds.vertices.length > 0) {
      ctx.beginPath();
      const first = ds.vertices[0];
      ctx.moveTo((first.x - c.viewX) * sx, (first.y - c.viewY) * sy);
      for (let i = 1; i < ds.vertices.length; i++) {
        ctx.lineTo((ds.vertices[i].x - c.viewX) * sx, (ds.vertices[i].y - c.viewY) * sy);
      }
      if (ds.cursorPos && ds.type === 'polygon') {
        ctx.lineTo((ds.cursorPos.x - c.viewX) * sx, (ds.cursorPos.y - c.viewY) * sy);
      }
      // Auto-close preview if near first vertex (polygon or freehand)
      if (ds.type === 'polygon' && ds.vertices.length >= 3 && ds.cursorPos) {
        const dist = Math.hypot(ds.cursorPos.x - first.x, ds.cursorPos.y - first.y);
        if (dist < 80) {
          ctx.lineTo((first.x - c.viewX) * sx, (first.y - c.viewY) * sy);
        }
      }
      if (ds.type === 'freehand' && ds.vertices.length >= 5) {
        const lastV = ds.vertices[ds.vertices.length - 1];
        const distF = Math.hypot(lastV.x - first.x, lastV.y - first.y);
        if (distF < 80) {
          ctx.lineTo((first.x - c.viewX) * sx, (first.y - c.viewY) * sy);
        }
      }
      ctx.stroke();
      if (ds.vertices.length > 2) {
        ctx.closePath();
        ctx.fill();
      }
      // Draw vertices for polygon
      if (ds.type === 'polygon') {
        for (let i = 0; i < ds.vertices.length; i++) {
          const v = ds.vertices[i];
          const isFirst = i === 0 && ds.vertices.length >= 3;
          ctx.beginPath();
          ctx.arc((v.x - c.viewX) * sx, (v.y - c.viewY) * sy, isFirst ? 5 : 3, 0, Math.PI * 2);
          ctx.fillStyle = isFirst ? 'rgba(220, 38, 38, 1)' : 'rgba(220, 38, 38, 0.7)';
          ctx.fill();
          // Snap indicator on first vertex
          if (isFirst && ds.cursorPos) {
            const dist = Math.hypot(ds.cursorPos.x - v.x, ds.cursorPos.y - v.y);
            if (dist < 80) {
              ctx.beginPath();
              ctx.arc((v.x - c.viewX) * sx, (v.y - c.viewY) * sy, 8, 0, Math.PI * 2);
              ctx.strokeStyle = 'rgba(220, 38, 38, 0.5)';
              ctx.setLineDash([]);
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }
          }
        }
      }
    }

    ctx.restore();
  }

  // ============================================================
  // SECTION 5: LEGEND
  // ============================================================

  function updateLegend(c) {
    // Legend removed — celltypes are shown in the controls panel
  }

  // ============================================================
  // SECTION 6: INTERACTION HANDLERS (ZOOM / PAN / ROI)
  // ============================================================

  function setupCanvasInteraction() {
    const el = document.getElementById('spatial-canvas');
    const container = document.getElementById('canvas-container');

    // Wheel zoom
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;

      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      const newWidth = c.viewWidth / factor;
      const newHeight = c.viewHeight / factor;
      const newZoom = c.imageWidth / newWidth;
      if (newZoom < MIN_ZOOM || newZoom > MAX_ZOOM) return;

      c.viewX = imgPos.x - mx * (newWidth / el.width);
      c.viewY = imgPos.y - my * (newHeight / el.height);
      c.viewWidth = newWidth;
      c.viewHeight = newHeight;
      c.zoom = newZoom;

      render();
      updateZoomDisplay();
    }, { passive: false });

    // Mouse down
    el.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;

      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      if (state.roiTool === 'bbox') {
        state.roiDrawingState = { type: 'bbox', start: imgPos, current: imgPos };
        return;
      }

      if (state.roiTool === 'freehand') {
        state.roiDrawingState = { type: 'freehand', vertices: [imgPos] };
        return;
      }

      // Pan
      if (!state.roiTool) {
        state.isPanning = true;
        state.panStart = { mx, my, vx: c.viewX, vy: c.viewY };
        container.classList.add('panning');
      }
    });

    // Mouse move
    el.addEventListener('mousemove', (e) => {
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      // Coordinate display
      document.getElementById('cursor-coords').textContent =
        `(${Math.round(imgPos.x)}, ${Math.round(imgPos.y)})`;

      if (state.isPanning && state.panStart) {
        const dx = (mx - state.panStart.mx) * c.viewWidth / el.width;
        const dy = (my - state.panStart.my) * c.viewHeight / el.height;
        c.viewX = state.panStart.vx - dx;
        c.viewY = state.panStart.vy - dy;
        render();
        return;
      }

      if (state.roiDrawingState) {
        if (state.roiDrawingState.type === 'bbox') {
          state.roiDrawingState.current = imgPos;
          render();
        } else if (state.roiDrawingState.type === 'polygon') {
          state.roiDrawingState.cursorPos = imgPos;
          render();
        } else if (state.roiDrawingState.type === 'freehand') {
          // Sample points while dragging (throttle to avoid too many)
          const verts = state.roiDrawingState.vertices;
          const last = verts[verts.length - 1];
          const dist = Math.hypot(imgPos.x - last.x, imgPos.y - last.y);
          if (dist > 3) { // min distance between points
            verts.push(imgPos);
            render();
          }
        }
      }
    });

    // Mouse up
    el.addEventListener('mouseup', (e) => {
      if (state.isPanning) {
        state.isPanning = false;
        state.panStart = null;
        container.classList.remove('panning');
        return;
      }

      // Complete bbox ROI
      if (state.roiDrawingState && state.roiDrawingState.type === 'bbox') {
        const ds = state.roiDrawingState;
        const minX = Math.min(ds.start.x, ds.current.x);
        const minY = Math.min(ds.start.y, ds.current.y);
        const maxX = Math.max(ds.start.x, ds.current.x);
        const maxY = Math.max(ds.start.y, ds.current.y);
        if (Math.abs(maxX - minX) > 5 && Math.abs(maxY - minY) > 5) {
          completeROI('bbox', { min_x: minX, min_y: minY, max_x: maxX, max_y: maxY });
        }
        state.roiDrawingState = null;
        render();
      }

      // Complete freehand ROI on mouse up
      if (state.roiDrawingState && state.roiDrawingState.type === 'freehand') {
        const ds = state.roiDrawingState;
        if (ds.vertices.length >= 5) {
          // Simplify path: take every Nth point to reduce vertex count
          const step = Math.max(1, Math.floor(ds.vertices.length / 100));
          const simplified = [];
          for (let i = 0; i < ds.vertices.length; i += step) {
            simplified.push([ds.vertices[i].x, ds.vertices[i].y]);
          }
          // Ensure last point is included
          const lastV = ds.vertices[ds.vertices.length - 1];
          const lastS = simplified[simplified.length - 1];
          if (lastS[0] !== lastV.x || lastS[1] !== lastV.y) {
            simplified.push([lastV.x, lastV.y]);
          }
          completeROI('polygon', { vertices: simplified });
        }
        state.roiDrawingState = null;
        render();
      }
    });

    // Click for polygon
    el.addEventListener('click', (e) => {
      if (state.roiTool !== 'polygon') return;
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      const rect = el.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.width / rect.width);
      const my = (e.clientY - rect.top) * (el.height / rect.height);
      const imgPos = canvasToImage(mx, my);

      if (!state.roiDrawingState) {
        state.roiDrawingState = { type: 'polygon', vertices: [imgPos], cursorPos: null };
      } else {
        // Check if close to first vertex (complete polygon)
        const first = state.roiDrawingState.vertices[0];
        const dist = Math.hypot(imgPos.x - first.x, imgPos.y - first.y);
        if (state.roiDrawingState.vertices.length >= 3 && dist < 80) {
          const verts = state.roiDrawingState.vertices.map(v => [v.x, v.y]);
          completeROI('polygon', { vertices: verts });
          state.roiDrawingState = null;
          render();
        } else {
          state.roiDrawingState.vertices.push(imgPos);
        }
      }
      render();
    });

    // Double-click completes polygon
    el.addEventListener('dblclick', (e) => {
      if (state.roiTool === 'polygon' && state.roiDrawingState &&
          state.roiDrawingState.vertices.length >= 3) {
        e.preventDefault();
        const verts = state.roiDrawingState.vertices.map(v => [v.x, v.y]);
        completeROI('polygon', { vertices: verts });
        state.roiDrawingState = null;
        render();
      }
    });
  }

  async function completeROI(type, params) {
    const name = prompt('ROI name:');
    if (!name) return;
    try {
      const result = await api.addROI(name, type, params);
      if (result.success) {
        await refreshROIs();
        render();
      } else {
        alert('Failed to create ROI: ' + (result.error || 'Unknown error'));
      }
    } catch (err) {
      alert('Error creating ROI: ' + err.message);
    }
    cancelROITool();
  }

  function cancelROITool() {
    state.roiTool = null;
    state.roiDrawingState = null;
    document.getElementById('canvas-container').classList.remove('roi-drawing');
    document.getElementById('roi-bbox-btn').classList.remove('active');
    document.getElementById('roi-polygon-btn').classList.remove('active');
    document.getElementById('roi-freehand-btn').classList.remove('active');
    document.getElementById('roi-cancel-btn').classList.add('hidden');
    render();
  }

  function resetView() {
    const c = getCurrentCanvas();
    const el = document.getElementById('spatial-canvas');
    if (!c || !c.imageWidth || !el) return;
    const aspect = el.width / el.height;
    const imgAspect = c.imageWidth / c.imageHeight;
    if (imgAspect > aspect) {
      c.viewWidth = c.imageWidth;
      c.viewHeight = c.imageWidth / aspect;
      c.viewX = 0;
      c.viewY = -(c.viewHeight - c.imageHeight) / 2;
    } else {
      c.viewHeight = c.imageHeight;
      c.viewWidth = c.imageHeight * aspect;
      c.viewX = -(c.viewWidth - c.imageWidth) / 2;
      c.viewY = 0;
    }
    c.zoom = el.width / c.viewWidth;
    render();
    updateZoomDisplay();
  }

  function updateZoomDisplay() {
    const c = getCurrentCanvas();
    document.getElementById('zoom-level').textContent =
      c ? `${Math.round(c.zoom * 100)}%` : '100%';
  }

  // ============================================================
  // SECTION 7: SSE CHAT ENGINE
  // ============================================================

  async function startChatStream(message) {
    state.chatStreaming = true;
    state.chatAbortController = new AbortController();
    _resetStreamState();
    updateChatButtons();

    // Add user message to UI
    appendUserMessage(message);

    // Create assistant message container
    const msgEl = createAssistantMessage();
    state.currentAssistantMsg = msgEl;

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
        signal: state.chatAbortController.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              handleSSEEvent(data);
            } catch (e) {
              // Ignore malformed JSON
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        appendToAssistantMsg('<div class="alert-card error">Connection error: ' + escapeHtml(err.message) + '</div>');
      }
    }

    state.chatStreaming = false;
    // Keep currentAssistantMsg alive — clarification replies continue in same bubble.
    // It gets replaced when the next startChatStream creates a new assistant message.
    updateChatButtons();
  }

  // Track what's been rendered to prevent duplicates across the event stream.
  // The backend emits the same plots/text in multiple events:
  //   agent_text → code_block_complete → execution_complete → done
  // We show content from the earliest event and skip repeats.
  let _shownPlotKeys = new Set();
  let _shownTextSnippets = new Set();
  let _hasExecutionIssue = false;
  let _execOutputEl = null;

  function _resetStreamState() {
    _shownPlotKeys = new Set();
    _shownTextSnippets = new Set();
    _hasExecutionIssue = false;
    _execOutputEl = null;
  }

  function _deduplicatePlots(plots) {
    if (!plots || !plots.length) return [];
    const novel = [];
    for (const p of plots) {
      const src = typeof p === 'string' ? p : (p.url || p);
      if (!src) continue;
      // Use first 100 chars of base64 as fingerprint
      const key = src.slice(0, 100);
      if (!_shownPlotKeys.has(key)) {
        _shownPlotKeys.add(key);
        novel.push(p);
      }
    }
    return novel;
  }

  function _isTextDuplicate(text) {
    if (!text) return true;
    // Use first 80 chars as fingerprint (handles repeated analysis text)
    const key = text.trim().slice(0, 80);
    if (!key) return true;
    if (_shownTextSnippets.has(key)) return true;
    _shownTextSnippets.add(key);
    return false;
  }

  function handleSSEEvent(event) {
    switch (event.type) {
      case 'pipeline_log':
        appendPipelineLog(event.message);
        break;
      case 'planning_complete':
        renderPlanCard(event.plan || event.steps);
        break;
      case 'step_start':
        highlightPlanStep(event.step_number);
        appendToAssistantMsg(`<div class="small-info" style="margin:6px 0;font-weight:600">Step ${event.step_number}${event.total_steps ? '/' + event.total_steps : ''}: ${escapeHtml(event.description || '')}</div>`);
        break;
      case 'skill_selection':
        renderSkillSelection(event);
        break;
      case 'clarification_needed':
        renderClarification(event.question, event.options);
        break;
      case 'prerequisites_needed':
        renderPrerequisites(event.questions, event.skill);
        break;
      case 'advice':
        appendToAssistantMsg(`<div class="alert-card advice">${renderMarkdown(event.message)}</div>`);
        break;
      case 'warning':
        appendToAssistantMsg(`<div class="alert-card warning">${renderMarkdown(event.message)}</div>`);
        break;
      case 'execution_start':
        // Create a collapsible output block for upcoming stdout
        _execOutputEl = null;
        break;
      case 'execution_output':
        if (event.line != null) {
          const ln = event.line.trim();
          // Skip progress bars (tqdm, keras, etc.) and empty lines
          if (/\d+%\|[█▓▒░▌▍▎▏ ]*\|/.test(ln)) break;
          if (/^\d+\/\d+\s+\[=+>?\.*\]/.test(ln)) break;
          if (!ln && event.stream === 'stdout') break;

          // Lazily create the output container
          if (!_execOutputEl) {
            if (!state.currentAssistantMsg) break;
            const details = document.createElement('details');
            details.className = 'exec-details';
            const summary = document.createElement('summary');
            summary.textContent = 'Output';
            details.appendChild(summary);
            const pre = document.createElement('div');
            pre.className = 'exec-output';
            details.appendChild(pre);
            _execOutputEl = pre;
            state.currentAssistantMsg.appendChild(details);
          }
          const lineEl = document.createElement('div');
          if (event.stream === 'stderr') lineEl.className = 'stderr';
          lineEl.textContent = event.line;
          _execOutputEl.appendChild(lineEl);
        }
        break;
      case 'agent_text':
        if (!_isTextDuplicate(event.text)) {
          _execOutputEl = null;  // Reset so next code block gets its own output section
          appendAgentTextFormatted(event.text);
        }
        break;
      case 'code_block_complete':
        _execOutputEl = null;  // Reset after code block completes
        renderPlots(_deduplicatePlots(event.plots));
        break;
      case 'execution_issue':
        _hasExecutionIssue = true;
        if (event.explanation) {
          appendToAssistantMsg(`<div class="alert-card error"><strong>Execution Issue</strong> (${escapeHtml(event.issue_type || 'error')})<br>${renderMarkdown(event.explanation)}</div>`);
        }
        break;
      case 'execution_complete':
      case 'step_execution_complete':
        // Only render plots — the text content was already shown via agent_text events.
        // event.response is for memory/logging, not for display.
        renderPlots(_deduplicatePlots(event.plots));
        if (event.step_number) markPlanStepDone(event.step_number);
        break;
      case 'reflection_start':
        appendToAssistantMsg(`<div class="small-info" style="color:var(--warning)">Reflecting on error (attempt ${event.attempt})...</div>`);
        break;
      case 'reflection_complete':
        break;
      case 'state_changes':
        handleStateChanges(event.changes);
        break;
      case 'done':
        renderPlots(_deduplicatePlots(event.plots));
        if (event.state_changes) handleStateChanges(event.state_changes);
        if (event.message && !_isTextDuplicate(event.message)) appendAgentTextFormatted(event.message);
        break;
      case 'error':
        appendToAssistantMsg(`<div class="alert-card error">${escapeHtml(event.message)}</div>`);
        break;
      default:
        break;
    }
    scrollChatToBottom();
  }

  // --- Chat UI helpers ---

  function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg user';
    msgDiv.innerHTML = `<div class="chat-msg-body">${escapeHtml(text)}</div>`;
    document.getElementById('chat-messages').appendChild(msgDiv);
    scrollChatToBottom();
  }

  function createAssistantMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg assistant';
    const body = document.createElement('div');
    body.className = 'chat-msg-body';
    msgDiv.appendChild(body);
    document.getElementById('chat-messages').appendChild(msgDiv);
    // Remove welcome message
    const welcome = document.querySelector('.chat-welcome');
    if (welcome) welcome.remove();
    return body;
  }

  function appendToAssistantMsg(html) {
    if (!state.currentAssistantMsg) return;
    state.currentAssistantMsg.insertAdjacentHTML('beforeend', html);
  }

  function appendAgentText(text) {
    if (!text) return;
    appendToAssistantMsg(`<div>${renderMarkdown(text)}</div>`);
  }

  // Formatted version: detects **Analysis:** blocks and renders them with special styling.
  // Also skips analysis text if there was an execution_issue (only the issue is shown).
  function appendAgentTextFormatted(text) {
    if (!text) return;

    // Detect analysis block from the _analyze_execution_results second LLM call
    const analysisMatch = text.match(/^\*\*Analysis:?\*\*\s*\n?([\s\S]*)$/);
    if (analysisMatch) {
      // If there was an execution issue, skip the analysis — issue UI is already shown
      if (_hasExecutionIssue) return;
      // Render analysis with separator and styled header
      const analysisBody = analysisMatch[1].trim();
      if (!analysisBody) return; // empty analysis — skip entirely
      appendToAssistantMsg(`<hr style="border:none;border-top:1px solid var(--border-light);margin:10px 0"><div><em><strong>Analysis</strong></em></div><div>${renderMarkdown(analysisBody)}</div>`);
      return;
    }

    appendAgentText(text);
  }

  function appendCodeBlock(code) {
    appendToAssistantMsg(`<div class="chat-code-block"><pre>${escapeHtml(code)}</pre></div>`);
  }

  function ensureExecOutput() {
    if (!state.currentAssistantMsg) return;
    let el = state.currentAssistantMsg.querySelector('.exec-output:last-of-type');
    if (!el) {
      el = document.createElement('div');
      el.className = 'exec-output';
      state.currentAssistantMsg.appendChild(el);
    }
  }

  function appendExecLine(line, stream) {
    if (!state.currentAssistantMsg) return;
    let el = state.currentAssistantMsg.querySelector('.exec-output:last-of-type');
    if (!el) {
      el = document.createElement('div');
      el.className = 'exec-output';
      state.currentAssistantMsg.appendChild(el);
    }
    if (stream === 'stderr') {
      el.insertAdjacentHTML('beforeend', `<span class="stderr">${escapeHtml(line || '')}\n</span>`);
    } else {
      el.insertAdjacentHTML('beforeend', escapeHtml(line || '') + '\n');
    }
    el.scrollTop = el.scrollHeight;
  }

  function closeExecOutput() {
    // Nothing special needed - just stops appending
  }

  function renderPlots(plots) {
    if (!plots || !state.currentAssistantMsg) return;
    for (const plot of plots) {
      const src = typeof plot === 'string' ? plot : (plot.url || plot);
      if (!src) continue;
      const imgSrc = src.startsWith('data:') ? src : `data:image/png;base64,${src}`;
      appendToAssistantMsg(`<div class="chat-plot"><img src="${imgSrc}" alt="Plot"></div>`);
    }
  }

  function appendPipelineLog(msg) {
    if (!state.currentAssistantMsg) return;
    let log = state.currentAssistantMsg.querySelector('.pipeline-log');
    if (!log) {
      log = document.createElement('details');
      log.className = 'pipeline-log';
      log.innerHTML = '<summary>Pipeline details</summary><div class="pipeline-log-content"></div>';
      state.currentAssistantMsg.appendChild(log);
    }
    log.querySelector('.pipeline-log-content').textContent += msg + '\n';
  }

  function renderPlanCard(plan) {
    if (!state.currentAssistantMsg) return;
    const steps = Array.isArray(plan) ? plan : (plan && plan.steps ? plan.steps : []);
    if (steps.length === 0) return;
    const stepsHtml = steps.map((s, i) => {
      const num = s.step_number || (i + 1);
      const desc = s.description || '';
      return `<li class="plan-step" data-step="${num}"><span class="plan-step-num">${num}</span>${escapeHtml(desc)}</li>`;
    }).join('');
    appendToAssistantMsg(`<details class="plan-card" open><summary>Plan (${steps.length} steps)</summary><ol class="plan-steps">${stepsHtml}</ol></details>`);
  }

  function highlightPlanStep(stepNum) {
    if (!state.currentAssistantMsg) return;
    const steps = state.currentAssistantMsg.querySelectorAll('.plan-step');
    steps.forEach(s => {
      s.classList.remove('active');
      if (parseInt(s.dataset.step) === stepNum) s.classList.add('active');
    });
  }

  function markPlanStepDone(stepNum) {
    if (!state.currentAssistantMsg) return;
    const steps = state.currentAssistantMsg.querySelectorAll('.plan-step');
    steps.forEach(s => {
      if (parseInt(s.dataset.step) === stepNum) {
        s.classList.remove('active');
        s.classList.add('done');
      }
    });
  }

  // Send a clarification reply inline — no new user/assistant bubbles.
  // The response continues in the same assistant message.
  function sendClarificationReply(text) {
    if (!text || !text.trim()) return;
    // Show what the user picked as a small inline note
    // Send via SSE — reuse the current assistant msg container
    _streamClarificationReply(text.trim());
  }

  async function _streamClarificationReply(message) {
    state.chatStreaming = true;
    state.chatAbortController = new AbortController();
    updateChatButtons();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
        signal: state.chatAbortController.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { handleSSEEvent(JSON.parse(line.slice(6))); } catch (e) {}
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        appendToAssistantMsg('<div class="alert-card error">Connection error: ' + escapeHtml(err.message) + '</div>');
      }
    }

    state.chatStreaming = false;
    updateChatButtons();
  }

  function renderSkillSelection(event) {
    if (!state.currentAssistantMsg) return;
    const options = event.options || [];
    const msg = event.message || 'Multiple skills matched. Please select:';
    const optHtml = options.map((o, i) =>
      `<button class="clarification-option-btn" data-value="${i + 1}">${i + 1}. ${escapeHtml(o.name || o.slug)}</button>`
    ).join('');
    appendToAssistantMsg(`<div class="clarification-ui"><div class="question">${escapeHtml(msg)}</div><div class="clarification-options">${optHtml}</div></div>`);

    setTimeout(() => {
      state.currentAssistantMsg.querySelectorAll('.clarification-option-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const val = btn.dataset.value;
          // Disable all buttons after selection
          btn.closest('.clarification-options').querySelectorAll('button').forEach(b => b.disabled = true);
          btn.classList.add('active');
          sendClarificationReply(val);
        });
      });
    }, 0);
  }

  function renderClarification(question, options) {
    if (!state.currentAssistantMsg) return;
    let html = `<div class="clarification-ui"><div class="question">${renderMarkdown(question || '')}</div>`;
    if (options && options.length) {
      html += '<div class="clarification-options">';
      options.forEach((o, i) => {
        const label = typeof o === 'string' ? o : (o.label || o.name || o);
        html += `<button class="clarification-option-btn" data-value="${escapeHtml(label)}">${escapeHtml(label)}</button>`;
      });
      html += '</div>';
    } else {
      // Free-text input for clarification
      html += '<div class="clarification-options"><input type="text" class="clarification-text-input" placeholder="Type your answer..." style="width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:var(--radius);font-size:12px"><button class="btn-secondary clarification-submit" style="margin-top:6px">Submit</button></div>';
    }
    html += '</div>';
    appendToAssistantMsg(html);

    setTimeout(() => {
      // Option buttons
      state.currentAssistantMsg.querySelectorAll('.clarification-option-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const val = btn.dataset.value;
          btn.closest('.clarification-options').querySelectorAll('button').forEach(b => b.disabled = true);
          btn.classList.add('active');
          sendClarificationReply(val);
        });
      });
      // Free-text submit
      const submitBtn = state.currentAssistantMsg.querySelector('.clarification-submit');
      if (submitBtn) {
        const input = submitBtn.previousElementSibling;
        submitBtn.addEventListener('click', () => {
          if (input.value.trim()) {
            submitBtn.disabled = true;
            input.disabled = true;
            sendClarificationReply(input.value);
          }
        });
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && input.value.trim()) {
            submitBtn.disabled = true;
            input.disabled = true;
            sendClarificationReply(input.value);
          }
        });
      }
    }, 0);
  }

  function renderPrerequisites(questions, skill) {
    if (!state.currentAssistantMsg || !questions || !questions.length) return;
    let html = `<div class="clarification-ui"><div class="question">Prerequisites needed for <strong>${escapeHtml(skill || 'skill')}</strong>:</div>`;
    html += '<div class="clarification-options">';
    questions.forEach((q, i) => {
      html += `<div style="margin:4px 0"><label style="font-size:12px;display:block;margin-bottom:2px">${escapeHtml(q)}</label><input type="text" class="prereq-input" data-idx="${i}" style="width:100%;padding:4px 6px;border:1px solid var(--border);border-radius:4px;font-size:12px"></div>`;
    });
    html += `<button class="btn-secondary prereq-submit" style="margin-top:8px">Submit</button>`;
    html += '</div></div>';
    appendToAssistantMsg(html);

    setTimeout(() => {
      const submitBtn = state.currentAssistantMsg.querySelector('.prereq-submit');
      if (submitBtn) {
        submitBtn.addEventListener('click', () => {
          const inputs = submitBtn.closest('.clarification-ui').querySelectorAll('.prereq-input');
          const answers = Array.from(inputs).map(inp => inp.value).join(', ');
          submitBtn.disabled = true;
          inputs.forEach(inp => inp.disabled = true);
          sendClarificationReply(answers);
        });
      }
    }, 0);
  }

  function renderMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    // Code fences
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<div class="chat-code-block"><pre>$2</pre></div>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Headings (must be before bold, at start of line)
    html = html.replace(/(^|\n)######\s+(.+)/g, '$1<h6 class="chat-heading">$2</h6>');
    html = html.replace(/(^|\n)#####\s+(.+)/g, '$1<h5 class="chat-heading">$2</h5>');
    html = html.replace(/(^|\n)####\s+(.+)/g, '$1<h4 class="chat-heading">$2</h4>');
    html = html.replace(/(^|\n)###\s+(.+)/g, '$1<h3 class="chat-heading">$2</h3>');
    html = html.replace(/(^|\n)##\s+(.+)/g, '$1<h2 class="chat-heading">$2</h2>');
    html = html.replace(/(^|\n)#\s+(.+)/g, '$1<h1 class="chat-heading">$2</h1>');
    // Horizontal rule
    html = html.replace(/(^|\n)---+(\n|$)/g, '$1<hr class="chat-hr">$2');
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    // Tables
    html = html.replace(/((?:^|\n)\|.+\|(?:\n\|.+\|)+)/g, function(tableBlock) {
      const rows = tableBlock.trim().split('\n').filter(r => r.trim());
      if (rows.length < 2) return tableBlock;
      let table = '<table class="chat-table">';
      rows.forEach((row, i) => {
        if (/^\|[\s\-:]+\|/.test(row.replace(/<br>/g, ''))) return;
        const cells = row.split('|').filter((_, j, a) => j > 0 && j < a.length - 1);
        const tag = i === 0 ? 'th' : 'td';
        table += '<tr>' + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
      });
      table += '</table>';
      return table;
    });
    // Blockquotes
    html = html.replace(/((?:^|\n)&gt; .+(?:\n&gt; .+)*)/g, function(block) {
      const content = block.trim().split('\n').map(l => l.replace(/^&gt; /, '')).join('<br>');
      return `<blockquote class="chat-blockquote">${content}</blockquote>`;
    });
    // Unordered lists
    html = html.replace(/((?:^|\n)[*\-] .+(?:\n[*\-] .+)*)/g, function(block) {
      const items = block.trim().split('\n').map(l => l.replace(/^[*\-] /, ''));
      return '<ul>' + items.map(i => `<li>${i}</li>`).join('') + '</ul>';
    });
    // Numbered lists
    html = html.replace(/((?:^|\n)\d+\. .+(?:\n\d+\. .+)*)/g, function(block) {
      const items = block.trim().split('\n').map(l => l.replace(/^\d+\. /, ''));
      return '<ol>' + items.map(i => `<li>${i}</li>`).join('') + '</ol>';
    });
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
  }

  function scrollChatToBottom() {
    const el = document.getElementById('chat-messages');
    el.scrollTop = el.scrollHeight;
  }

  function updateChatButtons() {
    document.getElementById('send-btn').classList.toggle('hidden', state.chatStreaming);
    document.getElementById('abort-btn').classList.toggle('hidden', !state.chatStreaming);
    document.getElementById('chat-input').disabled = state.chatStreaming;
  }

  function sendChatMessage(text) {
    if (!text || !text.trim()) return;
    startChatStream(text.trim());
  }

  // ============================================================
  // SECTION 8: DATA LOADING & STATE CHANGES
  // ============================================================

  async function loadSliceData(sliceId, modality) {
    const key = getCanvasKey(sliceId, modality);
    console.log(`loadSliceData: ${key}`);
    let c = state.canvases[key];
    if (!c) {
      c = createEmptyCanvas();
      state.canvases[key] = c;
    }
    if (c.loaded || c.loading) { console.log(`  skip: loaded=${c.loaded} loading=${c.loading}`); return; }
    c.loading = true;

    const isCurrentSlice = sliceId === state.currentSliceId && modality === state.currentModality;
    if (isCurrentSlice) showSpinner('Loading slice data...');

    try {
      console.time('fetch-data');
      // Fetch image and cells in parallel
      const [imgRes, cellRes] = await Promise.all([
        api.getImageData(sliceId, modality),
        api.getCellOverlay(sliceId, modality),
      ]);
      console.timeEnd('fetch-data');

      // Process image
      console.time('decode-image');
      if (imgRes.success && imgRes.image_data) {
        c.imageWidth = imgRes.image_width;
        c.imageHeight = imgRes.image_height;
        c.isScatter = imgRes.is_scatter || false;

        if (c.isScatter) {
          // Scatter-only (no real tissue image) — skip decoding the blank white image.
          // render() will fill white background directly on canvas.
          c.image = null;
          console.log(`Scatter mode: skip image decode (${c.imageWidth}x${c.imageHeight})`);
        } else {
          // Real tissue image — decode via blob for speed
          const binary = atob(imgRes.image_data);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          const blob = new Blob([bytes], { type: 'image/jpeg' });

          if (typeof createImageBitmap === 'function') {
            c.image = await createImageBitmap(blob);
          } else {
            const img = new Image();
            const url = URL.createObjectURL(blob);
            await new Promise((resolve, reject) => {
              img.onload = () => { URL.revokeObjectURL(url); resolve(); };
              img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Image decode failed')); };
              img.src = url;
            });
            c.image = img;
          }
        }
      }
      console.timeEnd('decode-image');

      // Process cells
      console.time('process-cells');
      if (cellRes.success) {
        c.cells = cellRes.cells;
        c.nCells = cellRes.n_cells || 0;
        c.hasCelltype = cellRes.has_celltype || false;
        c.celltypes = cellRes.celltypes || [];
        const backendColors = cellRes.celltype_colors || {};
        // Register backend colors in global registry, then normalize canvas to use global
        for (const [ct, color] of Object.entries(backendColors)) {
          if (!_globalCelltypeColors[ct]) _globalCelltypeColors[ct] = color;
        }
        // Always use global colors so all slices are consistent
        c.celltypeColors = {};
        for (const ct of (cellRes.celltypes || [])) {
          c.celltypeColors[ct] = getOrAssignColor(ct, backendColors);
        }
        c.selectedCelltypes = new Set(); // Empty = show all (no filtering)
        c.isSpotData = cellRes.is_spot_data || false;
        c.spotInfo = cellRes.spot_info || null;
      }
      console.timeEnd('process-cells');

      c.loaded = true;
    } catch (err) {
      console.error('Error loading slice data:', err);
    }

    c.loading = false;
    if (isCurrentSlice) {
      hideSpinner();
      resetView();
    }
  }

  async function switchSlice(sliceId, modality) {
    // Update local state first so getCurrentCanvas() returns the right one
    state.currentSliceId = sliceId;
    if (modality) state.currentModality = modality;

    // Notify backend
    api.selectSlice(sliceId).catch(() => {});
    if (modality) api.selectModality(modality).catch(() => {});

    // Load data if needed
    const c = getCurrentCanvas();
    if (!c.loaded && !c.loading) {
      await loadSliceData(sliceId, state.currentModality);
    }

    // Always render and sync UI
    if (c.loaded) {
      // If view was never initialized (preloaded in background), do it now
      if (!c.viewWidth) resetView();
      else { render(); updateZoomDisplay(); }
    }

    updateSliceTabs();
    updateModalityToggle();
    updateCelltypeCheckboxes();
    updateDataInfo();
    updateVizModeState();
    updateROIList();
    if (view3d.active) render3DView();
  }

  async function preloadAllSlices() {
    if (!state.sessionSummary) return;
    const slices = state.sessionSummary.available_slices || [];
    for (const s of slices) {
      const key = getCanvasKey(s.slice_id, s.modality);
      if (!state.canvases[key] || !state.canvases[key].loaded) {
        // Load in background (no await, just fire)
        loadSliceData(s.slice_id, s.modality);
      }
    }
  }

  async function refreshROIs() {
    try {
      const res = await api.listROIs();
      if (res.success) {
        state.rois = res.rois || [];
        updateROIList();
      }
    } catch (e) { /* ignore */ }
  }

  function handleStateChanges(changes) {
    if (!changes) return;
    const affected = new Set();
    (changes.celltypes_updated || []).forEach(sid => affected.add(sid));
    (changes.deconv_weights_updated || []).forEach(sid => affected.add(sid));
    (changes.celltype_colors_updated || []).forEach(sid => affected.add(sid));

    // Invalidate affected caches
    for (const sid of affected) {
      const mods = state.sessionSummary ? state.sessionSummary.modalities || ['gene'] : ['gene'];
      for (const mod of mods) {
        const key = getCanvasKey(sid, mod);
        if (state.canvases[key]) {
          state.canvases[key].loaded = false;
          state.canvases[key].loading = false;
        }
      }
    }

    // Reload current if affected
    if (affected.has(state.currentSliceId)) {
      loadSliceData(state.currentSliceId, state.currentModality).then(() => {
        render();
        updateCelltypeCheckboxes();
        updateDataInfo();
        if (view3d.active) render3DView();
      });
    }

    // Refresh ROIs if changed
    if (changes.rois_added || changes.rois_deleted) {
      refreshROIs().then(render);
    }
  }

  // ============================================================
  // SECTION 9: UI COMPONENT CONTROLLERS
  // ============================================================

  function updateSliceTabs() {
    const container = document.getElementById('slice-tabs');
    if (!state.sessionSummary) { container.innerHTML = ''; return; }
    const slices = state.sessionSummary.available_slices || [];
    // Group by modality - show each unique slice_id
    const seen = new Set();
    const uniqueSlices = slices.filter(s => {
      if (seen.has(s.slice_id)) return false;
      seen.add(s.slice_id);
      return true;
    });

    container.innerHTML = uniqueSlices.map(s =>
      `<button class="slice-tab ${s.slice_id === state.currentSliceId ? 'active' : ''}" data-slice-id="${s.slice_id}" data-modality="${s.modality}">${escapeHtml(s.tissue_name || 'Slice ' + s.slice_id)}</button>`
    ).join('');

    container.querySelectorAll('.slice-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        const sid = parseInt(btn.dataset.sliceId);
        const mod = btn.dataset.modality || state.currentModality;
        if (sid !== state.currentSliceId) switchSlice(sid, mod);
      });
    });
  }

  function updateModalityToggle() {
    const group = document.getElementById('modality-group');
    if (!state.sessionSummary) return;
    const hasProtein = state.sessionSummary.has_protein;
    group.classList.toggle('hidden', !hasProtein);

    document.querySelectorAll('#modality-toggle .seg-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.modality === state.currentModality);
    });
  }

  function updateVizModeState() {
    const c = getCurrentCanvas();
    const vm = c ? c.vizMode : 'celltype';
    const propRadio = document.querySelector('#proportion-option input');
    if (propRadio) {
      propRadio.disabled = !(c && c.isSpotData && c.spotInfo && c.spotInfo.has_deconv_weights);
    }
    // Sync radio buttons to per-canvas vizMode
    document.querySelectorAll('input[name="viz-mode"]').forEach(r => {
      r.checked = r.value === vm;
    });
    // Show/hide gene search
    document.getElementById('gene-search-group').classList.toggle('hidden', vm !== 'gene');
    // Sync gene search input
    document.getElementById('gene-search').value = (c && c.selectedGene) || '';
    // Show/hide proportion celltype selector
    const showPropCt = vm === 'proportion' && c && c.isSpotData && c.spotInfo && c.spotInfo.deconv_weights;
    document.getElementById('proportion-ct-group').classList.toggle('hidden', !showPropCt);
    if (showPropCt) updateProportionCtSelect();
    // Show/hide colorbar for gene and proportion modes
    const showColorbar = vm === 'gene' || (vm === 'proportion' && c && c.proportionCelltype);
    document.getElementById('colorbar-group').classList.toggle('hidden', !showColorbar);
    if (showColorbar) updateColorbar();
    // Sync opacity and point size sliders
    updateOverlaySliders();
  }

  function updateOverlaySliders() {
    const c = getCurrentCanvas();
    if (!c) return;
    document.getElementById('opacity-slider').value = Math.round(c.opacity * 100);
    document.getElementById('size-slider').value = Math.round(c.pointSize * 100);
    document.getElementById('hide-bg-cb').checked = c.hideBackground;
  }

  function updateProportionCtSelect() {
    const c = getCurrentCanvas();
    const select = document.getElementById('proportion-ct-select');
    if (!c || !c.spotInfo || !c.spotInfo.deconv_weights) return;

    const order = c.spotInfo.celltype_order || Object.keys(c.spotInfo.deconv_weights);
    const sorted = [...order].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
    select.innerHTML = '<option value="">Pie Chart (all types)</option>' +
      sorted.map(ct => `<option value="${escapeHtml(ct)}" ${ct === c.proportionCelltype ? 'selected' : ''}>${escapeHtml(ct)}</option>`).join('');
  }

  function updateColorbar() {
    const c = getCurrentCanvas();
    if (!c) return;

    const bar = document.getElementById('colorbar-bar');
    const minLabel = document.getElementById('colorbar-min');
    const maxLabel = document.getElementById('colorbar-max');
    const slider = document.getElementById('vmax-slider');
    const valueLabel = document.getElementById('vmax-value');

    if (!VIRIDIS_LUT) buildViridisLUT();

    if (c.vizMode === 'gene' && c.geneExpressionRange) {
      const [emin, rawMax] = c.geneExpressionRange;
      const emax = c.customVmax != null ? c.customVmax : rawMax;

      // Build viridis gradient for the bar
      const stops = [0, 0.25, 0.5, 0.75, 1].map(t =>
        VIRIDIS_LUT[Math.round(t * 255)]
      ).join(', ');
      bar.style.background = `linear-gradient(to right, ${stops})`;

      minLabel.textContent = emin.toFixed(2);
      maxLabel.textContent = emax.toFixed(2);

      // Set slider range: 1% to 100% of rawMax
      slider.min = 1;
      slider.max = 100;
      slider.value = c.customVmax != null ? Math.round((c.customVmax / rawMax) * 100) : 100;
      valueLabel.textContent = c.customVmax != null ? emax.toFixed(2) : 'auto';
    } else if (c.vizMode === 'proportion') {
      const stops = [0, 0.25, 0.5, 0.75, 1].map(t =>
        VIRIDIS_LUT[Math.round(t * 255)]
      ).join(', ');
      if (c.proportionCelltype) {
        // Single celltype heatmap — tunable vmax (0-1 range)
        const vmax = c.customVmax != null ? c.customVmax : 1.0;
        bar.style.background = `linear-gradient(to right, ${stops})`;
        minLabel.textContent = '0';
        maxLabel.textContent = vmax.toFixed(2);
        slider.disabled = false;
        slider.min = 1;
        slider.max = 100;
        slider.value = c.customVmax != null ? Math.round(c.customVmax * 100) : 100;
        valueLabel.textContent = c.customVmax != null ? vmax.toFixed(2) : 'auto';
      } else {
        // Pie chart mode — no vmax control
        bar.style.background = 'var(--bg-muted)';
        minLabel.textContent = '';
        maxLabel.textContent = 'Pie chart mode';
        slider.disabled = true;
        slider.value = 100;
        valueLabel.textContent = '';
      }
    } else {
      slider.disabled = false;
    }
  }

  function updateCelltypeCheckboxes() {
    const c = getCurrentCanvas();
    const container = document.getElementById('celltype-checkboxes');
    const noMsg = document.getElementById('no-celltype-msg');

    // Build full celltype list: merge obs celltypes + deconv weight celltypes
    let allCelltypes = c ? [...c.celltypes] : [];
    if (c && c.spotInfo && c.spotInfo.deconv_weights) {
      const deconvTypes = c.spotInfo.celltype_order || Object.keys(c.spotInfo.deconv_weights);
      for (const ct of deconvTypes) {
        if (!allCelltypes.includes(ct)) allCelltypes.push(ct);
      }
    }
    allCelltypes.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));

    if (!c || allCelltypes.length === 0) {
      container.innerHTML = '';
      noMsg.classList.remove('hidden');
      return;
    }
    noMsg.classList.add('hidden');

    // Ensure all celltypes have consistent colors via global registry
    for (const ct of allCelltypes) {
      c.celltypeColors[ct] = getOrAssignColor(ct, c.celltypeColors);
    }

    container.innerHTML = allCelltypes.map(ct => {
      const color = c.celltypeColors[ct];
      // Empty set = all shown (no filtering active)
      const checked = (c.selectedCelltypes.size === 0 || c.selectedCelltypes.has(ct)) ? 'checked' : '';
      return `<label class="ct-checkbox"><input type="checkbox" ${checked} data-ct="${escapeHtml(ct)}"><span class="ct-swatch" style="background:${color}"></span>${escapeHtml(ct)}</label>`;
    }).join('');

    const allCts = allCelltypes; // closure capture
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.addEventListener('change', () => {
        const ct = cb.dataset.ct;
        if (c.selectedCelltypes.size === 0 && !cb.checked) {
          allCts.forEach(t => c.selectedCelltypes.add(t));
        }
        if (cb.checked) c.selectedCelltypes.add(ct);
        else c.selectedCelltypes.delete(ct);
        render();
      });
    });

    // Add resize handle if not present
    let resizeHandle = container.parentElement.querySelector('.ct-resize-handle');
    if (!resizeHandle) {
      resizeHandle = document.createElement('div');
      resizeHandle.className = 'ct-resize-handle';
      container.after(resizeHandle);
      setupCtResizeHandle(resizeHandle, container);
    }
  }

  function setupCtResizeHandle(handle, list) {
    let startY, startH;
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      handle.classList.add('active');
      startY = e.clientY;
      startH = list.offsetHeight;
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';

      function onMove(e) {
        const delta = e.clientY - startY;
        const newH = Math.max(40, Math.min(600, startH + delta));
        list.style.maxHeight = newH + 'px';
      }
      function onUp() {
        handle.classList.remove('active');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  function getSliceName(sliceId) {
    if (!state.sessionSummary) return 'Slice ' + sliceId;
    const s = (state.sessionSummary.available_slices || []).find(s => s.slice_id === sliceId);
    return s ? (s.tissue_name || 'Slice ' + sliceId) : 'Slice ' + sliceId;
  }

  function updateROIList() {
    const container = document.getElementById('roi-list');
    const allRois = state.rois;
    if (allRois.length === 0) {
      container.innerHTML = '<div class="small-info">No ROIs defined</div>';
      return;
    }

    // Show all ROIs — active slice ones first, others dimmed
    const currentRois = [];
    const otherRois = [];
    for (const r of allRois) {
      if (r.slice_id === state.currentSliceId && r.modality === state.currentModality) {
        currentRois.push(r);
      } else {
        otherRois.push(r);
      }
    }

    function roiItemHtml(r, isActive) {
      const sliceLabel = getSliceName(r.slice_id);
      const cls = isActive ? 'roi-item' : 'roi-item roi-inactive';
      return `<div class="${cls}" data-roi="${escapeHtml(r.name)}" data-active="${isActive}">
        <div class="roi-item-main">
          <span class="roi-item-name">${escapeHtml(r.name)}</span>
          <span class="roi-item-slice">${escapeHtml(sliceLabel)}</span>
        </div>
        <div class="roi-item-detail hidden">${r.type} &middot; ${r.n_cells || '?'} cells</div>
        <span class="roi-item-actions">
          <button class="roi-action-btn roi-rename-btn" title="Rename">&#9998;</button>
          <button class="roi-action-btn roi-delete-btn" title="Delete">&times;</button>
        </span>
      </div>`;
    }

    container.innerHTML =
      currentRois.map(r => roiItemHtml(r, true)).join('') +
      otherRois.map(r => roiItemHtml(r, false)).join('');

    // Event listeners
    container.querySelectorAll('.roi-item').forEach(item => {
      const name = item.dataset.roi;
      const isActive = item.dataset.active === 'true';

      // Hover: highlight on canvas (only active slice) + show detail
      item.addEventListener('mouseenter', () => {
        item.querySelector('.roi-item-detail').classList.remove('hidden');
        if (isActive) {
          state.highlightedROI = name;
          item.classList.add('highlighted');
          render();
        }
      });
      item.addEventListener('mouseleave', () => {
        item.querySelector('.roi-item-detail').classList.add('hidden');
        if (isActive) {
          state.highlightedROI = null;
          item.classList.remove('highlighted');
          render();
        }
      });

      // Rename
      item.querySelector('.roi-rename-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        const newName = prompt('Rename ROI:', name);
        if (newName && newName !== name) {
          const res = await api.renameROI(name, newName);
          if (res.success) {
            await refreshROIs();
            render();
          } else {
            alert(res.error || 'Rename failed');
          }
        }
      });

      // Delete
      item.querySelector('.roi-delete-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm(`Delete ROI "${name}"?`)) {
          const res = await api.deleteROI(name);
          if (res.success) {
            state.highlightedROI = null;
            await refreshROIs();
            render();
          } else {
            alert(res.error || 'Delete failed');
          }
        }
      });
    });
  }

  function updateDataInfo() {
    const c = getCurrentCanvas();
    const el = document.getElementById('data-info');
    if (!c || !c.loaded) { el.textContent = ''; return; }
    const parts = [];
    parts.push(`${c.nCells.toLocaleString()} cells`);
    if (c.isSpotData) parts.push('Spot data');
    if (c.hasCelltype) parts.push(`${c.celltypes.length} types`);
    el.textContent = parts.join(' \u00b7 ');
  }

  function updateSessionInfo() {
    const el = document.getElementById('session-info');
    if (!state.sessionSummary) return;
    const s = state.sessionSummary;
    const parts = [s.name || 'Session'];
    if (s.n_slices > 1) parts.push(`${s.n_slices} slices`);
    if (s.n_cells) parts.push(`${s.n_cells.toLocaleString()} cells`);
    if (s.n_genes) parts.push(`${s.n_genes.toLocaleString()} genes`);
    el.textContent = parts.join(' \u00b7 ');
  }

  function setupGeneSearch() {
    const input = document.getElementById('gene-search');
    const dropdown = document.getElementById('gene-autocomplete');
    let activeIdx = -1;
    // Pre-sort gene list alphabetically once loaded
    let sortedGenes = null;

    function getSortedGenes() {
      if (!sortedGenes || sortedGenes.length !== state.geneList.length) {
        sortedGenes = [...state.geneList].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
      }
      return sortedGenes;
    }

    function showDropdown(query) {
      const genes = getSortedGenes();
      const q = (query || '').trim().toLowerCase();
      let matches;
      if (!q) {
        // Empty input: show all genes alphabetically (first 30)
        matches = genes.slice(0, 30);
      } else {
        // Filter: prefix matches first, then substring matches
        const prefix = [], substring = [];
        for (const g of genes) {
          const gl = g.toLowerCase();
          if (gl.startsWith(q)) prefix.push(g);
          else if (gl.includes(q)) substring.push(g);
          if (prefix.length + substring.length >= 30) break;
        }
        matches = [...prefix, ...substring].slice(0, 30);
      }

      if (matches.length === 0) { dropdown.classList.add('hidden'); return; }
      activeIdx = -1;
      dropdown.innerHTML = matches.map(g =>
        `<div class="autocomplete-item" data-gene="${escapeHtml(g)}">${escapeHtml(g)}</div>`
      ).join('');
      dropdown.classList.remove('hidden');

      dropdown.querySelectorAll('.autocomplete-item').forEach(item => {
        item.addEventListener('click', () => selectGene(item.dataset.gene));
      });
    }

    input.addEventListener('input', debounce(() => showDropdown(input.value), 100));

    // Show all genes on focus
    input.addEventListener('focus', () => showDropdown(input.value));

    input.addEventListener('keydown', (e) => {
      const items = dropdown.querySelectorAll('.autocomplete-item');
      if (!items.length) return;
      if (e.key === 'ArrowDown') { e.preventDefault(); activeIdx = Math.min(activeIdx + 1, items.length - 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); activeIdx = Math.max(activeIdx - 1, 0); }
      else if (e.key === 'Enter' && activeIdx >= 0) { e.preventDefault(); selectGene(items[activeIdx].dataset.gene); return; }
      else return;
      items.forEach((it, i) => it.classList.toggle('active', i === activeIdx));
      // Scroll active item into view
      if (items[activeIdx]) items[activeIdx].scrollIntoView({ block: 'nearest' });
    });

    document.addEventListener('click', (e) => {
      if (!e.target.closest('#gene-search-group')) dropdown.classList.add('hidden');
    });
  }

  async function selectGene(gene) {
    document.getElementById('gene-autocomplete').classList.add('hidden');
    document.getElementById('gene-search').value = gene;
    const c = getCurrentCanvas();
    if (c) { c.selectedGene = gene; c.vizMode = 'gene'; }
    document.querySelector('input[name="viz-mode"][value="gene"]').checked = true;
    updateVizModeState();

    showSpinner('Loading gene expression...');
    try {
      const res = await api.getGeneExpression(gene);
      if (res.success) {
        const c = getCurrentCanvas();
        c.geneExpression = res.cells;
        c.geneExpressionRange = res.expression_range;
        c.customVmax = null; // Reset vmax for new gene
        document.getElementById('gene-info').textContent =
          `${res.n_cells} cells, range: ${res.expression_range[0].toFixed(2)} - ${res.expression_range[1].toFixed(2)}`;
      }
    } catch (err) {
      console.error('Gene expression error:', err);
    }
    hideSpinner();
    updateColorbar();
    render();
  }

  // ============================================================
  // SECTION 10: INITIALIZATION
  // ============================================================

  function setupResizeObserver() {
    const container = document.getElementById('canvas-container');
    const canvasEl = document.getElementById('spatial-canvas');
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      canvasEl.width = container.clientWidth * dpr;
      canvasEl.height = container.clientHeight * dpr;
      canvasEl.style.width = container.clientWidth + 'px';
      canvasEl.style.height = container.clientHeight + 'px';
      if (state.initialized) render();
    });
    ro.observe(container);
  }

  function setupEventListeners() {
    // Init form
    document.getElementById('init-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const form = e.target;
      const submitBtn = document.getElementById('init-submit-btn');
      const errorEl = document.getElementById('init-error');
      const loadingEl = document.getElementById('init-loading');

      errorEl.classList.add('hidden');
      loadingEl.classList.remove('hidden');
      submitBtn.disabled = true;

      const params = {
        dataset_dir: form.dataset_dir.value,
        session_name: form.session_name.value,
        api_key: form.api_key.value || undefined,
        model: form.model.value || undefined,
        base_url: form.base_url.value || undefined,
      };

      try {
        const res = await api.initDataset(params);
        if (res.success) {
          // Persist form values for next time
          saveFormToStorage();

          state.sessionSummary = res.summary;
          state.currentSliceId = res.summary.current_slice_id;
          state.currentModality = res.summary.current_modality || 'gene';
          state.initialized = true;

          // Hide overlay, show app
          document.getElementById('init-overlay').classList.add('hidden');
          document.getElementById('app').classList.remove('hidden');

          // Enable chat if agent is active
          document.getElementById('send-btn').disabled = !res.agent_active;
          document.getElementById('chat-input').placeholder = res.agent_active
            ? 'Ask about your data...'
            : 'Agent not available (no API key)';

          // Setup UI
          updateSessionInfo();
          updateSliceTabs();
          updateModalityToggle();
          updateVizModeState();

          // Trigger canvas resize + load data
          setupResizeObserver();
          await loadSliceData(state.currentSliceId, state.currentModality);

          // Update controls that depend on loaded data
          updateCelltypeCheckboxes();
          updateDataInfo();
          updateVizModeState();

          // Load gene list
          api.getGeneList().then(res => {
            if (res.success) state.geneList = res.genes || [];
          });

          // Load ROIs
          refreshROIs();

          // Preload other slices
          setTimeout(preloadAllSlices, 500);
        } else {
          errorEl.textContent = res.error || res.message || 'Failed to load dataset';
          errorEl.classList.remove('hidden');
        }
      } catch (err) {
        errorEl.textContent = 'Connection error: ' + err.message;
        errorEl.classList.remove('hidden');
      }

      loadingEl.classList.add('hidden');
      submitBtn.disabled = false;
    });

    // Test LLM
    document.getElementById('test-llm-btn').addEventListener('click', async () => {
      const resultEl = document.getElementById('test-llm-result');
      resultEl.textContent = 'Testing...';
      resultEl.style.color = 'var(--text-secondary)';
      try {
        const modelVal = document.getElementById('model').value;
        // Auto-detect provider from model prefix (e.g. "poe/Model" -> "poe")
        const provMatch = modelVal.match(/^(\w+)\//);
        const provider = provMatch ? provMatch[1] : 'openai';
        const res = await api.testLLM({
          provider: provider,
          api_key: document.getElementById('api-key').value,
          model: modelVal,
          base_url: document.getElementById('base-url').value,
        });
        if (res.success) {
          resultEl.textContent = 'Connected!';
          resultEl.style.color = 'var(--success)';
        } else {
          resultEl.textContent = res.error || 'Failed';
          resultEl.style.color = 'var(--danger)';
        }
      } catch (e) {
        resultEl.textContent = 'Error: ' + e.message;
        resultEl.style.color = 'var(--danger)';
      }
    });

    // Logout
    document.getElementById('logout-btn').addEventListener('click', () => logout());

    // Modality toggle
    document.querySelectorAll('#modality-toggle .seg-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const mod = btn.dataset.modality;
        if (mod !== state.currentModality) {
          state.currentModality = mod;
          updateModalityToggle();
          // Find a slice with this modality
          const slices = state.sessionSummary.available_slices || [];
          const match = slices.find(s => s.modality === mod);
          if (match) switchSlice(match.slice_id, mod);
        }
      });
    });

    // Viz mode
    document.querySelectorAll('input[name="viz-mode"]').forEach(radio => {
      radio.addEventListener('change', () => {
        const c = getCurrentCanvas();
        if (c) c.vizMode = radio.value;
        updateVizModeState();
        render();
      });
    });

    // Proportion celltype select
    document.getElementById('proportion-ct-select').addEventListener('change', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      const val = document.getElementById('proportion-ct-select').value;
      c.proportionCelltype = val || null;
      c.customVmax = null; // reset vmax when switching
      updateVizModeState();
      render();
    });

    // Vmax slider
    document.getElementById('vmax-slider').addEventListener('input', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      const pct = parseInt(document.getElementById('vmax-slider').value);

      if (c.vizMode === 'gene' && c.geneExpressionRange) {
        const [emin, rawMax] = c.geneExpressionRange;
        if (pct >= 100) {
          c.customVmax = null;
          document.getElementById('vmax-value').textContent = 'auto';
          document.getElementById('colorbar-max').textContent = rawMax.toFixed(2);
        } else {
          c.customVmax = rawMax * (pct / 100);
          document.getElementById('vmax-value').textContent = c.customVmax.toFixed(2);
          document.getElementById('colorbar-max').textContent = c.customVmax.toFixed(2);
        }
      } else if (c.vizMode === 'proportion' && c.proportionCelltype) {
        // Proportion heatmap: vmax is 0-1
        if (pct >= 100) {
          c.customVmax = null;
          document.getElementById('vmax-value').textContent = 'auto';
          document.getElementById('colorbar-max').textContent = '1.00';
        } else {
          c.customVmax = pct / 100;
          document.getElementById('vmax-value').textContent = c.customVmax.toFixed(2);
          document.getElementById('colorbar-max').textContent = c.customVmax.toFixed(2);
        }
      }
      render();
    });

    // Opacity slider
    document.getElementById('opacity-slider').addEventListener('input', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.opacity = parseInt(document.getElementById('opacity-slider').value) / 100;
      render();
    });

    // Point size slider
    document.getElementById('size-slider').addEventListener('input', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.pointSize = parseInt(document.getElementById('size-slider').value) / 100;
      render();
    });

    // Hide background checkbox
    document.getElementById('hide-bg-cb').addEventListener('change', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.hideBackground = document.getElementById('hide-bg-cb').checked;
      render();
    });

    // Celltype select all / none
    document.getElementById('ct-select-all').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      c.selectedCelltypes = new Set(); // Empty = show all
      updateCelltypeCheckboxes();
      render();
    });

    document.getElementById('ct-select-none').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c) return;
      // Use a special marker set with a dummy entry to distinguish from "show all" empty set
      c.selectedCelltypes = new Set(['__none__']);
      updateCelltypeCheckboxes();
      render();
    });

    // ROI tools
    document.getElementById('roi-bbox-btn').addEventListener('click', () => {
      if (state.roiTool === 'bbox') { cancelROITool(); return; }
      state.roiTool = 'bbox';
      state.roiDrawingState = null;
      document.getElementById('canvas-container').classList.add('roi-drawing');
      document.getElementById('roi-bbox-btn').classList.add('active');
      document.getElementById('roi-polygon-btn').classList.remove('active');
      document.getElementById('roi-freehand-btn').classList.remove('active');
      document.getElementById('roi-cancel-btn').classList.remove('hidden');
    });

    document.getElementById('roi-polygon-btn').addEventListener('click', () => {
      if (state.roiTool === 'polygon') { cancelROITool(); return; }
      state.roiTool = 'polygon';
      state.roiDrawingState = null;
      document.getElementById('canvas-container').classList.add('roi-drawing');
      document.getElementById('roi-polygon-btn').classList.add('active');
      document.getElementById('roi-bbox-btn').classList.remove('active');
      document.getElementById('roi-freehand-btn').classList.remove('active');
      document.getElementById('roi-cancel-btn').classList.remove('hidden');
    });

    document.getElementById('roi-freehand-btn').addEventListener('click', () => {
      if (state.roiTool === 'freehand') { cancelROITool(); return; }
      state.roiTool = 'freehand';
      state.roiDrawingState = null;
      document.getElementById('canvas-container').classList.add('roi-drawing');
      document.getElementById('roi-freehand-btn').classList.add('active');
      document.getElementById('roi-bbox-btn').classList.remove('active');
      document.getElementById('roi-polygon-btn').classList.remove('active');
      document.getElementById('roi-cancel-btn').classList.remove('hidden');
    });

    document.getElementById('roi-cancel-btn').addEventListener('click', cancelROITool);

    // Zoom buttons
    document.getElementById('zoom-in-btn').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      const el = document.getElementById('spatial-canvas');
      const centerX = c.viewX + c.viewWidth / 2;
      const centerY = c.viewY + c.viewHeight / 2;
      c.viewWidth /= ZOOM_FACTOR;
      c.viewHeight /= ZOOM_FACTOR;
      c.viewX = centerX - c.viewWidth / 2;
      c.viewY = centerY - c.viewHeight / 2;
      c.zoom = c.imageWidth / c.viewWidth;
      render();
      updateZoomDisplay();
    });

    document.getElementById('zoom-out-btn').addEventListener('click', () => {
      const c = getCurrentCanvas();
      if (!c || !c.loaded) return;
      c.viewWidth *= ZOOM_FACTOR;
      c.viewHeight *= ZOOM_FACTOR;
      const centerX = c.viewX + c.viewWidth / (2 * ZOOM_FACTOR);
      const centerY = c.viewY + c.viewHeight / (2 * ZOOM_FACTOR);
      c.viewX = centerX - c.viewWidth / 2;
      c.viewY = centerY - c.viewHeight / 2;
      c.zoom = c.imageWidth / c.viewWidth;
      render();
      updateZoomDisplay();
    });

    document.getElementById('zoom-fit-btn').addEventListener('click', resetView);

    // 3D toggle
    document.getElementById('toggle-3d-btn').addEventListener('click', toggle3DView);

    // Chat
    const chatInput = document.getElementById('chat-input');
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (text && !state.chatStreaming) {
          chatInput.value = '';
          sendChatMessage(text);
        }
      }
    });

    chatInput.addEventListener('input', () => {
      // Auto-resize
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });

    document.getElementById('send-btn').addEventListener('click', () => {
      const text = chatInput.value.trim();
      if (text && !state.chatStreaming) {
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendChatMessage(text);
      }
    });

    document.getElementById('abort-btn').addEventListener('click', () => {
      if (state.chatAbortController) state.chatAbortController.abort();
      api.chatAbort();
    });

    document.getElementById('save-chat-btn').addEventListener('click', () => {
      api.saveChatHistory();
    });

    // Notebook
    document.getElementById('notebook-btn').addEventListener('click', async () => {
      try {
        const res = await api.getNotebookURL();
        if (res.success) {
          window.open(`http://localhost:${res.jupyter_port}`, '_blank');
        } else {
          alert(res.error || 'Notebook not available');
        }
      } catch (e) {
        alert('Notebook not available');
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (state.roiTool) cancelROITool();
      }
    });
  }

  // ============================================================
  // SECTION: 3D VIEWER
  // ============================================================

  const view3d = {
    active: false,
    rotX: -30,
    rotY: 20,
    zoom: 1.0,
    spacing: 140,
    orbiting: false,
    dragMoved: false,    // distinguish click vs drag
    orbitStart: null,
    orbitStartRot: null,
  };

  function toggle3DView() {
    view3d.active = !view3d.active;
    const viewer3d = document.getElementById('viewer-3d');
    const canvas2d = document.getElementById('spatial-canvas');
    const btn = document.getElementById('toggle-3d-btn');

    if (view3d.active) {
      viewer3d.classList.remove('hidden');
      canvas2d.style.visibility = 'hidden';
      btn.classList.add('active');
      btn.textContent = '2D View';
      render3DView();
      setup3DInteraction();
    } else {
      viewer3d.classList.add('hidden');
      canvas2d.style.visibility = '';
      btn.classList.remove('active');
      btn.textContent = '3D View';
      teardown3DInteraction();
      render();
    }
  }

  function renderSliceToCanvas(sliceId, modality, displayW, displayH) {
    const key = getCanvasKey(sliceId, modality);
    const c = state.canvases[key];
    if (!c || !c.loaded) return null;

    // Render at full image resolution (capped at 2048 for perf), CSS scales down
    const imgW = c.imageWidth || displayW;
    const imgH = c.imageHeight || displayH;
    const maxDim = 2048;
    const resFactor = Math.min(maxDim / Math.max(imgW, imgH), 1);
    const renderW = Math.round(imgW * resFactor);
    const renderH = Math.round(imgH * resFactor);

    const offscreen = document.createElement('canvas');
    offscreen.width = renderW;
    offscreen.height = renderH;
    offscreen.style.width = displayW + 'px';
    offscreen.style.height = displayH + 'px';
    const ctx = offscreen.getContext('2d');

    // Draw image or white bg at full resolution
    if (c.image && !c.hideBackground) {
      ctx.drawImage(c.image, 0, 0, imgW, imgH, 0, 0, renderW, renderH);
    } else {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, renderW, renderH);
    }

    // Draw cells at full resolution
    const fakeEl = { width: renderW, height: renderH };
    const saved = { vx: c.viewX, vy: c.viewY, vw: c.viewWidth, vh: c.viewHeight };
    c.viewX = 0; c.viewY = 0;
    c.viewWidth = imgW;
    c.viewHeight = imgH;

    ctx.globalAlpha = c.opacity;
    if (c.vizMode === 'celltype') {
      renderCells(ctx, c, fakeEl);
    } else if (c.vizMode === 'gene' && c.geneExpression) {
      renderGeneExpression(ctx, c, fakeEl);
    } else if (c.vizMode === 'proportion' && c.isSpotData && c.spotInfo) {
      renderProportions(ctx, c, fakeEl);
    }
    ctx.globalAlpha = 1.0;

    // Restore view
    c.viewX = saved.vx; c.viewY = saved.vy;
    c.viewWidth = saved.vw; c.viewHeight = saved.vh;

    return offscreen;
  }

  function render3DView() {
    if (!view3d.active) return;

    const scene = document.getElementById('scene-3d');
    const container = document.getElementById('viewer-3d');
    const cw = container.clientWidth;
    const ch = container.clientHeight;

    // Gather all loaded slices
    const slices = (state.sessionSummary?.available_slices || []).filter(s => {
      const key = getCanvasKey(s.slice_id, s.modality);
      return state.canvases[key]?.loaded;
    });
    if (slices.length === 0) return;

    // Get per-slice dimensions, find max for uniform display width
    const sliceDims = slices.map(s => {
      const c = state.canvases[getCanvasKey(s.slice_id, s.modality)];
      return { w: c?.imageWidth || 400, h: c?.imageHeight || 400 };
    });
    const maxW = Math.max(...sliceDims.map(d => d.w));
    const maxH = Math.max(...sliceDims.map(d => d.h));

    // Size planes to fill the view well:
    // - Horizontally: use up to 75% of container width
    // - Vertically: account for slice height + total stack depth projected at the view angle
    const stackDepth = (slices.length - 1) * view3d.spacing;
    const projectedStackH = maxH * (maxW / maxH) + Math.abs(Math.sin(view3d.rotX * Math.PI / 180)) * stackDepth;
    const fitScale = Math.min((cw * 0.75) / maxW, (ch * 0.7) / projectedStackH);
    const displayW = Math.round(maxW * fitScale);

    scene.innerHTML = '';

    // Stack slices vertically in Z
    const totalSpacing = (slices.length - 1) * view3d.spacing;
    const startZ = totalSpacing / 2;

    slices.forEach((s, i) => {
      const c = state.canvases[getCanvasKey(s.slice_id, s.modality)];
      if (!c) return;

      // Per-slice aspect ratio
      const sw = c.imageWidth || 400;
      const sh = c.imageHeight || 400;
      const planeW = displayW;
      const planeH = Math.round(displayW * (sh / sw));

      const offscreen = renderSliceToCanvas(s.slice_id, s.modality, planeW, planeH);
      if (!offscreen) return;

      const plane = document.createElement('div');
      plane.className = 'slice-plane' + (s.slice_id === state.currentSliceId ? ' active-slice' : '');
      plane.style.width = planeW + 'px';
      plane.style.height = planeH + 'px';
      plane.style.left = ((cw - planeW) / 2) + 'px';
      plane.style.top = ((ch - planeH) / 2) + 'px';

      const z = startZ - i * view3d.spacing;
      plane.style.transform = `translateZ(${z}px)`;

      plane.appendChild(offscreen);

      // Label
      const label = document.createElement('div');
      label.className = 'slice-label';
      label.textContent = `${s.tissue_name || 'Slice ' + s.slice_id} (${s.modality})`;
      plane.appendChild(label);

      // Click: switch to that slice in 2D view (only if not dragging)
      plane.addEventListener('click', () => {
        if (view3d.dragMoved) return;
        switchSlice(s.slice_id, s.modality);
        toggle3DView(); // Exit 3D, show selected slice in 2D
      });

      scene.appendChild(plane);
    });

    updateSceneTransform();
  }

  function updateSceneTransform() {
    const scene = document.getElementById('scene-3d');
    if (!scene) return;
    const container = document.getElementById('viewer-3d');
    const cx = container.clientWidth / 2;
    const cy = container.clientHeight / 2;
    scene.style.perspective = `${1200 / view3d.zoom}px`;
    scene.style.perspectiveOrigin = `${cx}px ${cy}px`;
    scene.style.transform = `scale(${view3d.zoom}) rotateX(${view3d.rotX}deg) rotateY(${view3d.rotY}deg)`;
  }

  let _3dHandlers = {};

  function setup3DInteraction() {
    const viewer = document.getElementById('viewer-3d');

    _3dHandlers.mousedown = (e) => {
      if (e.target.closest('.viewer-3d-controls')) return;
      view3d.orbiting = true;
      view3d.dragMoved = false;
      view3d.orbitStart = { x: e.clientX, y: e.clientY };
      view3d.orbitStartRot = { x: view3d.rotX, y: view3d.rotY };
      viewer.classList.add('orbiting');
      e.preventDefault();
    };

    _3dHandlers.mousemove = (e) => {
      if (!view3d.orbiting) return;
      const dx = e.clientX - view3d.orbitStart.x;
      const dy = e.clientY - view3d.orbitStart.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) view3d.dragMoved = true;
      view3d.rotY = view3d.orbitStartRot.y + dx * 0.3;
      // Allow full rotation — no clamping
      view3d.rotX = view3d.orbitStartRot.x - dy * 0.3;
      updateSceneTransform();
    };

    _3dHandlers.mouseup = () => {
      view3d.orbiting = false;
      viewer.classList.remove('orbiting');
    };

    _3dHandlers.wheel = (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.92 : 1.08;
      view3d.zoom = Math.max(0.2, Math.min(6, view3d.zoom * factor));
      updateSceneTransform();
    };

    _3dHandlers.spacing = () => {
      view3d.spacing = parseInt(document.getElementById('spacing-slider').value);
      render3DView();
    };

    viewer.addEventListener('mousedown', _3dHandlers.mousedown);
    document.addEventListener('mousemove', _3dHandlers.mousemove);
    document.addEventListener('mouseup', _3dHandlers.mouseup);
    viewer.addEventListener('wheel', _3dHandlers.wheel, { passive: false });
    document.getElementById('spacing-slider').addEventListener('input', _3dHandlers.spacing);
  }

  function teardown3DInteraction() {
    const viewer = document.getElementById('viewer-3d');
    if (!viewer) return;
    viewer.removeEventListener('mousedown', _3dHandlers.mousedown);
    document.removeEventListener('mousemove', _3dHandlers.mousemove);
    document.removeEventListener('mouseup', _3dHandlers.mouseup);
    viewer.removeEventListener('wheel', _3dHandlers.wheel);
    const spacingSlider = document.getElementById('spacing-slider');
    if (spacingSlider) spacingSlider.removeEventListener('input', _3dHandlers.spacing);
    _3dHandlers = {};
  }

  // ============================================================
  // SECTION: RESIZABLE PANELS
  // ============================================================

  function setupResizableHandles() {
    const main = document.querySelector('.main-layout');
    if (!main) return;

    function setupHandle(handleId, colIndex, minPx, maxPx) {
      const handle = document.getElementById(handleId);
      if (!handle) return;

      let startX, startSize, cols;

      handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        handle.classList.add('active');
        startX = e.clientX;
        cols = main.style.gridTemplateColumns
          ? main.style.gridTemplateColumns.split(/\s+/)
          : getComputedStyle(main).gridTemplateColumns.split(/\s+/);
        startSize = parseFloat(cols[colIndex]);
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        function onMove(e) {
          const delta = e.clientX - startX;
          // Both panels are to the RIGHT of their handle:
          // Dragging handle right = panel shrinks (negative direction)
          const direction = -1;
          let newSize = Math.max(minPx, Math.min(maxPx, startSize + delta * direction));
          cols[colIndex] = newSize + 'px';
          // Keep viewer as 1fr
          cols[0] = '1fr';
          main.style.gridTemplateColumns = cols.join(' ');
          // Trigger canvas resize
          if (state.initialized) {
            const container = document.getElementById('canvas-container');
            const canvasEl = document.getElementById('spatial-canvas');
            const dpr = window.devicePixelRatio || 1;
            canvasEl.width = container.clientWidth * dpr;
            canvasEl.height = container.clientHeight * dpr;
            render();
          }
        }

        function onUp() {
          handle.classList.remove('active');
          document.body.style.cursor = '';
          document.body.style.userSelect = '';
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
        }

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
      });
    }

    // Left handle controls the controls-panel width (column index 2)
    setupHandle('resize-left', 2, 150, 400);
    // Right handle controls the chat-panel width (column index 4)
    setupHandle('resize-right', 4, 300, 900);
  }

  async function init() {
    buildViridisLUT();
    setupEventListeners();
    setupCanvasInteraction();
    setupGeneSearch();
    setupResizableHandles();

    // Restore saved form values
    loadFormFromStorage();

    // If backend session is still alive, reconnect without re-init
    await tryReconnectSession();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
