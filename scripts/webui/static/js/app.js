import { postJSON } from './api.js';
import { ConfigEditor } from './config-editor.js';
import { TerminalView } from './terminal.js';
import { getRunOptions, setRunOptions } from './runner.js';

const defaults = window.__APP_DEFAULTS__ || {};

const projectPathInput = document.getElementById('projectPath');
const configPathInput = document.getElementById('configPath');
const statusEl = document.getElementById('projectStatus');
const participantsPicker = document.getElementById('participantsPicker');
const configParticipantsPicker = document.getElementById('configParticipantsPicker');
const groupColumnPicker = document.getElementById('groupColumnPicker');

const terminal = new TerminalView(document.getElementById('terminalConsole'));
const editor = new ConfigEditor(
  document.getElementById('configEditor'),
  document.getElementById('fieldCountBadge')
);

let currentProjectPath = defaults.defaultProject || 'projects/webui/project.json';

function currentConfigParticipantsPath() {
  const fromPicker = configParticipantsPicker?.value?.trim();
  if (fromPicker) return fromPicker;

  const fromConfig = editor.getValueAtPath(['analysis', 'participants_file']);
  if (typeof fromConfig === 'string' && fromConfig.trim()) return fromConfig.trim();

  const fromRunOption = document.getElementById('optParticipants')?.value?.trim();
  if (fromRunOption) return fromRunOption;

  return '';
}

function fillSelect(selectEl, values, selectedValue = '', placeholder = 'Select...') {
  if (!selectEl) return;
  selectEl.innerHTML = '';

  const first = document.createElement('option');
  first.value = '';
  first.textContent = placeholder;
  selectEl.appendChild(first);

  values.forEach((value) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    if (value === selectedValue) option.selected = true;
    selectEl.appendChild(option);
  });
}

async function refreshParticipantsDerivedFields() {
  const participantsPath = currentConfigParticipantsPath();
  const result = await postJSON('/api/participants/columns', { participants_path: participantsPath });
  const columns = result.columns || [];

  editor.setParticipantsColumns(columns);

  const currentGroupColumn = editor.getValueAtPath(['analysis', 'group_column']) || '';
  fillSelect(groupColumnPicker, columns, currentGroupColumn, 'Select group column...');
}

async function refreshParticipantsFilePicker() {
  const response = await fetch('/api/participants/files');
  const result = await response.json();
  if (!result.success) return;

  const files = result.files || [];
  const currentRun = document.getElementById('optParticipants')?.value?.trim() || '';
  const currentCfg = editor.getValueAtPath(['analysis', 'participants_file']) || '';

  fillSelect(participantsPicker, files, currentRun, 'Select a participants TSV...');
  fillSelect(configParticipantsPicker, files, currentCfg, 'Select a participants TSV...');
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
  await refreshParticipantsFilePicker();
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
    try {
      await refreshParticipantsFilePicker();
      await refreshParticipantsDerivedFields();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  configParticipantsPicker.addEventListener('change', async () => {
    const picked = configParticipantsPicker.value;
    if (!picked) return;

    editor.setValueAtPath(['analysis', 'participants_file'], picked);
    document.getElementById('optParticipants').value = picked;

    try {
      await refreshParticipantsFilePicker();
      await refreshParticipantsDerivedFields();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  groupColumnPicker.addEventListener('change', async () => {
    const selected = groupColumnPicker.value;
    if (!selected) return;
    editor.setValueAtPath(['analysis', 'group_column'], selected);
  });

  participantsPicker.addEventListener('change', async () => {
    const picked = participantsPicker.value;
    if (!picked) return;
    document.getElementById('optParticipants').value = picked;
    try {
      await refreshParticipantsDerivedFields();
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnRefreshParticipants').addEventListener('click', async () => {
    try {
      await refreshParticipantsFilePicker();
      await refreshParticipantsDerivedFields();
      setStatus('Participants file list refreshed', 'info');
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  document.getElementById('btnRefreshConfigParticipants').addEventListener('click', async () => {
    try {
      await refreshParticipantsFilePicker();
      await refreshParticipantsDerivedFields();
      setStatus('Config participants and columns refreshed', 'info');
    } catch (error) {
      setStatus(error.message, 'danger');
    }
  });

  editor.rootEl.addEventListener('change', async (event) => {
    const target = event.target;
    if (!target?.dataset?.path) return;
    if (target.dataset.path === JSON.stringify(['analysis', 'participants_file'])) {
      try {
        await refreshParticipantsFilePicker();
        await refreshParticipantsDerivedFields();
      } catch (error) {
        setStatus(error.message, 'danger');
      }
    }
    if (target.dataset.path === JSON.stringify(['analysis', 'group_column'])) {
      try {
        await refreshParticipantsDerivedFields();
      } catch (error) {
        setStatus(error.message, 'danger');
      }
    }
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
