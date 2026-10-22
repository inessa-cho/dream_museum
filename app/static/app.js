const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
const button = document.getElementById('generateButton');
const progressList = document.getElementById('progressList');
const modelMessage = document.getElementById('modelMessage');
const errorMessage = document.getElementById('errorMessage');
const galleryGrid = document.getElementById('galleryGrid');
const emptyGallery = document.getElementById('emptyGallery');
let lastGenerationId = null;

for (let i = 0; i < 10; i++) {
  const row = document.createElement('div');
  row.className = 'progress-item';
  row.innerHTML = `<span>${String(i + 1).padStart(2, '0')}</span><div class="progress-track"><div class="progress-fill"></div></div><span class="percent">0%</span>`;
  progressList.appendChild(row);
}

tabs.forEach(tab => tab.addEventListener('click', () => {
  tabs.forEach(t => t.classList.toggle('active', t === tab));
  panels.forEach(panel => panel.classList.toggle('active', panel.id === tab.dataset.tab));
}));

button.addEventListener('click', async () => {
  try {
    button.disabled = true;
    errorMessage.hidden = true;
    const response = await fetch('/api/generate', { method: 'POST' });
    if (!response.ok) throw new Error((await response.json()).detail || 'Could not start generation.');
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
    button.disabled = false;
  }
});

function updateProgress(state) {
  const rows = [...progressList.children];
  rows.forEach((row, index) => {
    const value = Math.max(0, Math.min(100, state.progress[index] || 0));
    row.querySelector('.progress-fill').style.width = `${value}%`;
    row.querySelector('.percent').textContent = `${Math.round(value)}%`;
    row.classList.toggle('complete', value >= 100);
  });
  button.disabled = state.running;
  button.textContent = state.running ? 'Generating…' : 'Generate';
  modelMessage.hidden = !state.model_loading;
  errorMessage.hidden = !state.error;
  if (state.error) errorMessage.textContent = state.error;
}

function updateGallery(state) {
  const hasImages = state.images.length > 0;
  emptyGallery.hidden = hasImages;
  if (state.generation_id === lastGenerationId && galleryGrid.children.length === state.images.length) return;
  lastGenerationId = state.generation_id;
  galleryGrid.innerHTML = '';
  state.images.forEach((src, index) => {
    const figure = document.createElement('figure');
    figure.className = 'artwork';
    figure.innerHTML = `<div class="frame"><div class="mat"><img src="${src}" alt="Dream Museum artwork ${index + 1}"></div></div>`;
    figure.style.animationDelay = `${Math.min(index, 9) * 60}ms`;
    galleryGrid.appendChild(figure);
  });
}

async function poll() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('Status unavailable');
    const state = await response.json();
    updateProgress(state);
    updateGallery(state);
  } catch (_) {}
  setTimeout(poll, 1000);
}
poll();
