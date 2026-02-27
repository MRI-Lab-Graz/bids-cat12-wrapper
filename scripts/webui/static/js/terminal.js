export class TerminalView {
  constructor(preEl) {
    this.preEl = preEl;
    this.eventSource = null;
  }

  clear() {
    this.preEl.textContent = '';
  }

  append(line) {
    this.preEl.textContent += `${line}\n`;
    this.preEl.scrollTop = this.preEl.scrollHeight;
  }

  stopStream() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  startStream(onDone) {
    this.stopStream();
    this.eventSource = new EventSource('/api/run/stream');

    this.eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'line') {
        this.append(data.text);
      }
      if (data.type === 'done') {
        this.append(`[done] exit_code=${data.exit_code}`);
        this.stopStream();
        if (onDone) onDone(data.exit_code);
      }
    };

    this.eventSource.onerror = () => {
      this.append('[stream disconnected]');
      this.stopStream();
    };
  }
}
