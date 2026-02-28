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
  const fromConfig = editor.getValueAtPath(['analysis', 'participants_file']);
  if (typeof fromConfig === 'string' && fromConfig.trim()) return fromConfig.trim();

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
  pathPickerUpBtn.disabled = pickerState.current === '';

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
  editor.setValueAtPath(['analysis', 'participants_file'], path);
  document.getElementById('optParticipants').value = path;
}

function setStatus(msg, kind = 'muted') {
  statusEl.className = `small text-${kind} mt-2`;
  statusEl.textContent = msg;
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

  const payload = {
    project_path: projectPathInput.value.trim(),
    config_data: editor.collect(),
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
      editor.setValueAtPath(['analysis', 'participants_file'], entered);
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

  document.getElementById('btnBrowseResultsDir').addEventListener('click', async () => {
    await openPathPicker({
      title: 'Pick Results Folder',
      allowFiles: false,
      allowDirs: true,
      startPath: document.getElementById('optResultsDir').value.trim() || 'results',
      onPick: (path) => {
        document.getElementById('optResultsDir').value = path;
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
    if (target.dataset.path === JSON.stringify(['analysis', 'participants_file'])) {
      try {
        const picked = editor.getValueAtPath(['analysis', 'participants_file']) || '';
        document.getElementById('optParticipants').value = String(picked);
        await refreshParticipantsDerivedFields();
      } catch (error) {
        setStatus(error.message, 'danger');
      }
    }
  });

  editor.rootEl.addEventListener('click', async (event) => {
    const target = event.target;
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

  document.getElementById('btnClearConsole').addEventListener('click', () => terminal.clear());
  document.getElementById('btnShutdown').addEventListener('click', shutdownApp);
}

async function init() {
  projectPathInput.value = currentProjectPath;
  await loadProject(currentProjectPath);
  bindEvents();
}

init().catch((error) => {
  setStatus(error.message, 'danger');
});
