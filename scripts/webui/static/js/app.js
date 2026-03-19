import { postJSON } from './api.js';
import { ConfigEditor } from './config-editor.js';
import { TerminalView } from './terminal.js';
import { getRunOptions, setRunOptions } from './runner.js';

const defaults = window.__APP_DEFAULTS__ || {};

const projectPathInput = document.getElementById('projectPath');
const configPathInput = document.getElementById('configPath');
const statusEl = document.getElementById('projectStatus');

const pathPickerModalEl = document.getElementById('pathPickerModal');
const pathPickerTitleEl = document.getElementById('pathPickerTitle');
const pathPickerCurrentEl = document.getElementById('pathPickerCurrent');
const pathPickerListEl = document.getElementById('pathPickerList');
const pathPickerUpBtn = document.getElementById('pathPickerUp');
const pathPickerSelectCurrentBtn = document.getElementById('pathPickerSelectCurrent');
const pathPickerModal = pathPickerModalEl ? new bootstrap.Modal(pathPickerModalEl) : null;

const terminal = new TerminalView(document.getElementById('terminalConsole'));
const editor = new ConfigEditor(
  document.getElementById('configEditor'),
  document.getElementById('fieldCountBadge')
);

let currentProjectPath = defaults.defaultProject || 'projects/webui/project.json';

const pickerState = {
  title: 'Browse',
  current: '',
  parent: '',
  allowFiles: true,
  allowDirs: false,
  extensions: [],
  onPick: null,
};

function currentConfigParticipantsPath() {
  const statsPath = editor.getValueAtPath(['statistics', 'input', 'participants_file']);
  if (typeof statsPath === 'string' && statsPath.trim()) return statsPath.trim();

  const legacyPath = editor.getValueAtPath(['analysis', 'participants_file']);
  if (typeof legacyPath === 'string' && legacyPath.trim()) return legacyPath.trim();

  const fromRunOption = document.getElementById('optParticipants')?.value?.trim();
  if (fromRunOption) return fromRunOption;

  return '';
}

function dirnamePath(path) {
  if (!path) return '';
  const normalized = String(path).replace(/\\/g, '/');
  const parts = normalized.split('/').filter(Boolean);
  if (parts.length <= 1) return '';
  parts.pop();
  return parts.join('/');
}

async function listFs(path, { allowFiles = true, allowDirs = true, extensions = [] } = {}) {
  const params = new URLSearchParams();
  params.set('path', path || '');
  params.set('files', allowFiles ? '1' : '0');
  params.set('dirs', allowDirs ? '1' : '0');
  if (extensions.length) {
    params.set('ext', extensions.join(','));
  }

  const response = await fetch(`/api/fs/list?${params.toString()}`);
  return response.json();
}

function renderPathPickerEntries(entries = []) {
  pathPickerListEl.innerHTML = '';

  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'text-muted small p-2';
    empty.textContent = 'No matching entries in this folder.';
    pathPickerListEl.appendChild(empty);
    return;
  }

  entries.forEach((entry) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'list-group-item list-group-item-action d-flex align-items-center gap-2';
    row.innerHTML = `${entry.is_dir ? '<i class="fas fa-folder text-warning"></i>' : '<i class="fas fa-file text-secondary"></i>'}<span>${entry.name}</span>`;

    row.addEventListener('click', async () => {
      if (entry.is_dir) {
        await openPathPickerAt(entry.path);
        return;
      }
      if (!pickerState.allowFiles) return;
      if (typeof pickerState.onPick === 'function') {
        pickerState.onPick(entry.path);
      }
      pathPickerModal.hide();
    });

    pathPickerListEl.appendChild(row);
  });
}

async function openPathPickerAt(path) {
  const result = await listFs(path, {
    allowFiles: pickerState.allowFiles,
    allowDirs: pickerState.allowDirs,
    extensions: pickerState.extensions,
  });

  if (!result.success) {
    setStatus('Failed to read workspace paths', 'danger');
    return;
  }

  pickerState.current = result.current || '';
  pickerState.parent = result.parent || '';

  pathPickerTitleEl.textContent = pickerState.title;
  pathPickerCurrentEl.textContent = pickerState.current || '.';
  pathPickerUpBtn.disabled = !pickerState.parent;

  if (pickerState.allowDirs) {
    pathPickerSelectCurrentBtn.classList.remove('d-none');
  } else {
    pathPickerSelectCurrentBtn.classList.add('d-none');
  }

  renderPathPickerEntries(result.entries || []);
}

async function openPathPicker(options) {
  pickerState.title = options.title || 'Browse';
  pickerState.allowFiles = options.allowFiles !== false;
  pickerState.allowDirs = Boolean(options.allowDirs);
  pickerState.extensions = options.extensions || [];
  pickerState.onPick = options.onPick || null;

  const start = options.startPath || '';
  await openPathPickerAt(start);
  pathPickerModal.show();
}

async function refreshParticipantsDerivedFields() {
  const participantsPath = currentConfigParticipantsPath();
  const result = await postJSON('/api/participants/columns', { participants_path: participantsPath });
  const columns = result.columns || [];

  editor.setParticipantsColumns(columns);
}

function applyParticipantsPath(path) {
  if (!path) return;
  if (editor.getValueAtPath(['statistics', 'input', 'participants_file']) !== undefined) {
    editor.setValueAtPath(['statistics', 'input', 'participants_file'], path);
  } else {
    editor.setValueAtPath(['analysis', 'participants_file'], path);
  }
  document.getElementById('optParticipants').value = path;
}

async function listReports() {
  const payload = {
    config_data: editor.collect(),
    run_options: getRunOptions(),
  };

  const result = await postJSON('/api/reports/list', payload);
  const select = document.getElementById('reportSelect');
  select.innerHTML = '';

  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '(no report selected)';
  select.appendChild(placeholder);

  const reports = result.reports || [];
  reports.forEach((reportPath) => {
    const option = document.createElement('option');
    option.value = reportPath;
    option.textContent = reportPath;
    select.appendChild(option);
  });

  // Auto-select newest report (backend returns newest-first when sorted by mtime)
  if (reports.length) {
    select.value = reports[0];
    setStatus(`Found ${reports.length} HTML report(s)`, 'success');
  } else {
    setStatus('No HTML reports found yet', 'warning');
  }
}

function openSelectedReport() {
  const selected = document.getElementById('reportSelect').value;
  if (!selected) {
    setStatus('Select a report first', 'warning');
    return;
  }
  const url = `/api/reports/open?path=${encodeURIComponent(selected)}`;
  window.open(url, '_blank', 'noopener');
}

async function generateReport() {
  terminal.clear();
  setStatus('Starting report generation...', 'info');

  const payload = {
    config_data: editor.collect(),
    results_dir: document.getElementById('optReportResultsDir').value.trim(),
    quality: document.getElementById('optReportQuality').value.trim(),
    report_filter: document.getElementById('optReportFilter').value.trim(),
    output_html: document.getElementById('optReportOutputHtml').value.trim(),
  };

  const result = await postJSON('/api/reports/generate', payload);
  if (!result.success) {
    setStatus(result.error || 'Failed to start report generation', 'danger');
    return;
  }

  terminal.startStream(async (exitCode) => {
    if (exitCode === 0) {
      setStatus('Report generated successfully', 'success');
      await listReports();
    } else {
      setStatus(`Report generation failed (exit ${exitCode})`, 'danger');
    }
  });
}

function setStatus(msg, kind = 'muted') {
  statusEl.className = `small text-${kind} mt-2`;
  statusEl.textContent = msg;
}

function normalizeOpenNeuroDatasetId(value) {
  return String(value || '').trim().toLowerCase();
}

function isValidOpenNeuroDatasetId(value) {
  return /^ds\d{6}$/i.test(normalizeOpenNeuroDatasetId(value));
}

function validateOpenNeuroConfig(configData) {
  const bidsCfg = configData?.preprocessing?.bids;
  if (!bidsCfg || !bidsCfg.openneuro) {
    return { ok: true };
  }

  const datasetId = normalizeOpenNeuroDatasetId(
    bidsCfg.openneuro_dataset || bidsCfg.bids_dir || ''
  );

  if (!datasetId) {
    return {
      ok: false,
      message: 'OpenNeuro is enabled but dataset ID is missing. Use format dsXXXXXX (e.g., ds004937).',
    };
  }

  if (!isValidOpenNeuroDatasetId(datasetId)) {
    return {
      ok: false,
      message: `Invalid OpenNeuro dataset ID "${datasetId}". Expected format: dsXXXXXX (e.g., ds004937).`,
    };
  }

  return { ok: true, datasetId };
}

function updateOpenNeuroInputValidation(inputEl) {
  if (!inputEl?.dataset?.bidsSourceField) return;

  const sourceMode = inputEl.dataset.sourceMode || 'local';
  if (sourceMode !== 'openneuro') {
    inputEl.classList.remove('is-valid', 'is-invalid');
    return;
  }

  const datasetId = normalizeOpenNeuroDatasetId(inputEl.value);
  if (!datasetId) {
    inputEl.classList.remove('is-valid', 'is-invalid');
    return;
  }

  if (isValidOpenNeuroDatasetId(datasetId)) {
    inputEl.classList.add('is-valid');
    inputEl.classList.remove('is-invalid');
  } else {
    inputEl.classList.add('is-invalid');
    inputEl.classList.remove('is-valid');
  }
}

async function loadProject(path) {
  const result = await postJSON('/api/project/load', { project_path: path });
  const project = result.project;
  currentProjectPath = project.project_path || path;

  projectPathInput.value = currentProjectPath;
  configPathInput.value = project.config_path || defaults.defaultConfig || '';
  editor.load(project.config_data || {});
  setRunOptions(project.run_options || {});
  await refreshParticipantsDerivedFields();
  setStatus(`Loaded ${currentProjectPath}`, 'success');
}

async function saveProject() {
  const payload = {
    project_path: projectPathInput.value.trim(),
    config_path: configPathInput.value.trim(),
    config_data: editor.collect(),
    run_options: getRunOptions()
  };
  const result = await postJSON('/api/project/save', payload);
  currentProjectPath = result.project_path;
  setStatus(`Saved ${currentProjectPath}`, 'success');
}

async function validateConfig() {
  const payload = {
    project_path: projectPathInput.value.trim(),
    config_data: editor.collect()
  };
  const result = await postJSON('/api/config/validate', payload);
  if (result.success) {
    setStatus('Config schema validation passed', 'success');
  } else {
    setStatus(`Validation failed: ${(result.errors || []).slice(0, 1).join('; ')}`, 'danger');
  }
}

async function runPipeline() {
  terminal.clear();
  setStatus('Starting pipeline...', 'info');

  const configData = editor.collect();
  const openNeuroValidation = validateOpenNeuroConfig(configData);
  if (!openNeuroValidation.ok) {
    setStatus(openNeuroValidation.message, 'danger');
    terminal.append(`[error] ${openNeuroValidation.message}`);
    return;
  }

  const payload = {
    project_path: projectPathInput.value.trim(),
    config_data: configData,
    run_options: getRunOptions()
  };

  const result = await postJSON('/api/run/start', payload);
  terminal.append(`$ ${result.command.join(' ')}`);
  terminal.startStream((exitCode) => {
    if (exitCode === 0) setStatus('Pipeline finished successfully', 'success');
    else setStatus(`Pipeline failed (exit ${exitCode})`, 'danger');
  });
}

async function stopPipeline() {
  await postJSON('/api/run/stop', {});
  setStatus('Stop signal sent', 'warning');
}

async function shutdownApp() {
  await fetch('/shutdown', { method: 'POST' });
}

function bindEvents() {
  document.getElementById('btnLoadProject').addEventListener('click', async () => {
    try {
      await loadProject(projectPathInput.value.trim());
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnSaveProject').addEventListener('click', async () => {
    try {
      await saveProject();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('optParticipants').addEventListener('change', async () => {
    const entered = document.getElementById('optParticipants').value.trim();
    if (entered) {
      if (editor.getValueAtPath(['statistics', 'input', 'participants_file']) !== undefined) {
        editor.setValueAtPath(['statistics', 'input', 'participants_file'], entered);
      } else {
        editor.setValueAtPath(['analysis', 'participants_file'], entered);
      }
    }
    try {
      await refreshParticipantsDerivedFields();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnBrowseProject').addEventListener('click', async () => {
    await openPathPicker({
      title: 'Pick Project File',
      allowFiles: true,
      allowDirs: true,
      extensions: ['.json'],
      startPath: dirnamePath(projectPathInput.value.trim()) || 'projects',
      onPick: (path) => {
        projectPathInput.value = path;
      },
    });
  });

  document.getElementById('btnBrowseConfig').addEventListener('click', async () => {
    await openPathPicker({
      title: 'Pick Config File',
      allowFiles: true,
      allowDirs: true,
      extensions: ['.json'],
      startPath: dirnamePath(configPathInput.value.trim()) || 'config',
      onPick: (path) => {
        configPathInput.value = path;
      },
    });
  });

  document.getElementById('btnBrowseParticipants').addEventListener('click', async () => {
    await openPathPicker({
      title: 'Pick Participants TSV',
      allowFiles: true,
      allowDirs: true,
      extensions: ['.tsv'],
      startPath: dirnamePath(document.getElementById('optParticipants').value.trim()) || 'results/data',
      onPick: async (path) => {
        applyParticipantsPath(path);
        await refreshParticipantsDerivedFields();
      },
    });
  });

  document.getElementById('btnBrowseCat12Dir').addEventListener('click', async () => {
    await openPathPicker({
      title: 'Pick CAT12 Folder',
      allowFiles: false,
      allowDirs: true,
      startPath: document.getElementById('optCat12Dir').value.trim() || '',
      onPick: (path) => {
        document.getElementById('optCat12Dir').value = path;
      },
    });
  });

  document.getElementById('btnBrowseStatsConfig').addEventListener('click', async () => {
    await openPathPicker({
      title: 'Pick Stats Config',
      allowFiles: true,
      allowDirs: true,
      extensions: ['.json'],
      startPath: dirnamePath(document.getElementById('optStatsConfig').value.trim()) || 'projects',
      onPick: (path) => {
        document.getElementById('optStatsConfig').value = path;
      },
    });
  });

  document.getElementById('btnBrowseReportResultsDir').addEventListener('click', async () => {
    await openPathPicker({
      title: 'Pick Report Input Folder',
      allowFiles: false,
      allowDirs: true,
      startPath: document.getElementById('optReportResultsDir').value.trim() || 'results',
      onPick: (path) => {
        document.getElementById('optReportResultsDir').value = path;
      },
    });
  });

  document.getElementById('btnRefreshParticipants').addEventListener('click', async () => {
    try {
      await refreshParticipantsDerivedFields();
      setStatus('Derived columns refreshed', 'info');
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  pathPickerUpBtn.addEventListener('click', async () => {
    await openPathPickerAt(pickerState.parent || '');
  });

  pathPickerSelectCurrentBtn.addEventListener('click', () => {
    if (typeof pickerState.onPick === 'function') {
      pickerState.onPick(pickerState.current || '');
    }
    pathPickerModal.hide();
  });

  editor.rootEl.addEventListener('change', async (event) => {
    const target = event.target;
    if (!target?.dataset?.path) return;

    if (target.dataset.bidsSourceField) {
      updateOpenNeuroInputValidation(target);
    }

    const statsParticipantsPath = JSON.stringify(['statistics', 'input', 'participants_file']);
    const legacyParticipantsPath = JSON.stringify(['analysis', 'participants_file']);

    if (target.dataset.path === statsParticipantsPath || target.dataset.path === legacyParticipantsPath) {
      try {
        const picked = currentConfigParticipantsPath();
        document.getElementById('optParticipants').value = String(picked);
        await refreshParticipantsDerivedFields();
      } catch (error) {
        setStatus(error.message, 'danger');
      }
    }
  });

  editor.rootEl.addEventListener('click', async (event) => {
    const target = event.target;

    if (target?.dataset?.bidsModeLocal) {
      const bidsTokens = JSON.parse(target.dataset.path || '[]');
      const draft = editor.collect();

      let cursor = draft;
      bidsTokens.forEach((token) => {
        if (!cursor[token] || typeof cursor[token] !== 'object') {
          cursor[token] = {};
        }
        cursor = cursor[token];
      });

      cursor.openneuro = false;
      cursor.openneuro_download = false;
      cursor.openneuro_download_only_anat = false;
      cursor.bids_dir = String(cursor.bids_dir || '').trim();

      editor.load(draft);
      setStatus('Input source set to local BIDS folder', 'info');
      return;
    }

    if (target?.dataset?.openneuroActivate) {
      const bidsTokens = JSON.parse(target.dataset.path || '[]');
      const draft = editor.collect();

      let cursor = draft;
      bidsTokens.forEach((token) => {
        if (!cursor[token] || typeof cursor[token] !== 'object') {
          cursor[token] = {};
        }
        cursor = cursor[token];
      });

      const datasetId = normalizeOpenNeuroDatasetId(
        cursor.openneuro_dataset || cursor.bids_dir || ''
      );
      if (!datasetId) {
        setStatus('Enter an OpenNeuro dataset ID first (e.g., ds004937)', 'warning');
        return;
      }
      if (!isValidOpenNeuroDatasetId(datasetId)) {
        setStatus(`Invalid dataset ID "${datasetId}". Use format dsXXXXXX (e.g., ds004937).`, 'warning');
        return;
      }

      cursor.openneuro = true;
      cursor.openneuro_dataset = datasetId;
      cursor.bids_dir = datasetId;
      cursor.openneuro_download = true;
      cursor.openneuro_download_only_anat = true;
      cursor.openneuro_download_all = false;
      cursor.source = `OpenNeuro ${datasetId}`;

      editor.load(draft);
      setStatus(`OpenNeuro dataset configured: ${datasetId}`, 'success');
      return;
    }

    if (target?.dataset?.folderPicker) {
      const tokens = JSON.parse(target.dataset.path || '[]');
      const currentFolder = editor.getValueAtPath(tokens) || '';

      await openPathPicker({
        title: target.dataset.folderPickerTitle || 'Pick Folder',
        allowFiles: false,
        allowDirs: true,
        startPath: String(currentFolder || '') || '.',
        onPick: (path) => {
          editor.setValueAtPath(tokens, path);
        },
      });
      return;
    }

    if (target?.dataset?.clearField) {
      const tokens = JSON.parse(target.dataset.path || '[]');
      editor.setValueAtPath(tokens, '');
      return;
    }

    if (!target?.dataset?.maskPicker) return;

    const tokens = JSON.parse(target.dataset.path || '[]');
    const currentMask = editor.getValueAtPath(tokens) || '';

    await openPathPicker({
      title: 'Pick Mask File',
      allowFiles: true,
      allowDirs: true,
      extensions: ['.nii', '.nii.gz', '.img', '.hdr'],
      startPath: dirnamePath(String(currentMask)) || 'templates',
      onPick: (path) => {
        editor.setValueAtPath(tokens, path);
      },
    });
  });

  editor.rootEl.addEventListener('input', (event) => {
    const target = event.target;
    if (!target?.dataset?.bidsSourceField) return;
    updateOpenNeuroInputValidation(target);
  });

  document.getElementById('btnValidate').addEventListener('click', async () => {
    try {
      await validateConfig();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnRun').addEventListener('click', async () => {
    try {
      await runPipeline();
    } catch (error) {
      terminal.append(`[error] ${error.message}`);
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnStop').addEventListener('click', async () => {
    try {
      await stopPipeline();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnListReports').addEventListener('click', async () => {
    try {
      await listReports();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnOpenReport').addEventListener('click', openSelectedReport);

  document.getElementById('btnClearConsole').addEventListener('click', () => terminal.clear());
  document.getElementById('btnShutdown').addEventListener('click', shutdownApp);

  document.getElementById('btnGenerateReport').addEventListener('click', async () => {
    try {
      await generateReport();
    } catch (error) {
      terminal.append(`[error] ${error.message}`);
      setStatus(error.message, 'danger');
    }
  });
}

async function init() {
  projectPathInput.value = currentProjectPath;
  await loadProject(currentProjectPath);
  bindEvents();
}

init().catch((error) => {
  setStatus(error.message, 'danger');
});
