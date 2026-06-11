/**
 * RAG MongoDB - Frontend Application
 * NotebookLM-inspired interface
 */

const API_BASE = '';

// DOM Elements
const themeToggle = document.getElementById('themeToggle');
const healthStatus = document.getElementById('healthStatus');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatStrategy = document.getElementById('chatStrategy');
const chatTopK = document.getElementById('chatTopK');
const quickSearchInput = document.getElementById('quickSearchInput');
const quickSearchStrategy = document.getElementById('quickSearchStrategy');
const quickSearchBtn = document.getElementById('quickSearchBtn');
const quickResults = document.getElementById('quickResults');
const refreshStatsBtn = document.getElementById('refreshStats');
const globalSearch = document.getElementById('globalSearch');
const loadingOverlay = document.getElementById('loadingOverlay');

// State
let isProcessing = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    checkHealth();
    loadStats();
    loadDocuments();
    loadImages();
    initEventListeners();
});

// Theme Management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
}

// Health Check
async function checkHealth() {
    const dot = healthStatus.querySelector('.health-dot');
    const text = healthStatus.querySelector('.health-text');

    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();

        if (data.status === 'ok') {
            dot.className = 'health-dot connected';
            text.textContent = 'Conectado';
        } else {
            dot.className = 'health-dot error';
            text.textContent = 'Error';
        }
    } catch (err) {
        dot.className = 'health-dot error';
        text.textContent = 'Sin conexión';
    }
}

// Load Stats
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();

        document.getElementById('totalDocs').textContent = data.total_documentos;

        let totalChunks = 0;
        const strategyCounts = {};

        data.chunks_por_estrategia.forEach(s => {
            totalChunks += s.total_chunks;
            strategyCounts[s.estrategia] = s.total_chunks;
        });

        document.getElementById('totalChunks').textContent = totalChunks;

        // Update strategy bars
        const maxChunks = Math.max(...Object.values(strategyCounts), 1);

        ['fixed', 'sentence-aware', 'semantic'].forEach(strategy => {
            const count = strategyCounts[strategy] || 0;
            const fill = document.querySelector(`.strategy-fill[data-strategy="${strategy}"]`);
            const countEl = document.getElementById(`count-${strategy}`);

            if (fill) {
                fill.style.width = `${(count / maxChunks) * 100}%`;
            }
            if (countEl) {
                countEl.textContent = count;
            }
        });
    } catch (err) {
        console.error('Error loading stats:', err);
    }
}

// Chat Functions
async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    showLoading(true);

    // Remove welcome message
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message
    addMessage(question, 'user');
    chatInput.value = '';
    autoResizeTextarea();

    try {
        const strategy = chatStrategy.value || null;
        const topK = parseInt(chatTopK.value);

        const res = await fetch(`${API_BASE}/rag`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pregunta: question,
                estrategia: strategy,
                top_k: topK
            })
        });

        const data = await res.json();
        if (res.ok) {
            addMessage(data.respuesta, 'assistant', data.contexto, data.estrategia_usada);
        } else {
            addMessage(`Error: ${data.detail || 'Error desconocido'}`, 'assistant');
        }
    } catch (err) {
        console.error('RAG Error:', err);
        addMessage('Error al procesar la pregunta. Verifica que el servidor esté ejecutándose.', 'assistant');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        showLoading(false);
    }
}

function addMessage(content, role, contexto = null, estrategia = null) {
    const div = document.createElement('div');
    div.className = `message message-${role}`;

    let html = `<div class="message-content"><p>${escapeHtml(content)}</p>`;

    if (role === 'assistant' && contexto && contexto.length > 0) {
        html += `<div class="message-sources">
            <h4>Fuentes consultadas</h4>
            <div class="source-chips">`;

        contexto.forEach((ctx, i) => {
            const score = ctx.score ? ctx.score.toFixed(3) : '-';
            html += `<span class="source-chip">
                Fragmento ${i + 1} | ${ctx.estrategia_chunking || estrategia || 'N/A'}
                <span class="score">${score}</span>
            </span>`;
        });

        html += `</div></div>`;
    }

    html += '</div>';
    div.innerHTML = html;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Quick Search
async function quickSearch() {
    const query = quickSearchInput.value.trim();
    if (!query) return;

    showLoading(true);

    try {
        const strategy = quickSearchStrategy.value || null;

        const res = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: 5,
                estrategia: strategy
            })
        });

        const data = await res.json();
        displayQuickResults(data.resultados);
    } catch (err) {
        quickResults.innerHTML = '<p style="color: var(--error); font-size: 0.8rem;">Error en la búsqueda</p>';
    } finally {
        showLoading(false);
    }
}

function displayQuickResults(resultados) {
    if (!resultados || resultados.length === 0) {
        quickResults.innerHTML = '<p style="color: var(--text-tertiary); font-size: 0.8rem;">No se encontraron resultados</p>';
        return;
    }

    quickResults.innerHTML = resultados.map(r => `
        <div class="quick-result-item">
            <div class="result-text">${escapeHtml(r.chunk_texto)}</div>
            <div class="result-meta">
                ${r.estrategia_chunking} | Score: ${r.score ? r.score.toFixed(3) : '-'}
            </div>
        </div>
    `).join('');
}

// Event Listeners
function initEventListeners() {
    themeToggle.addEventListener('click', toggleTheme);
    refreshStatsBtn.addEventListener('click', loadStats);
    sendBtn.addEventListener('click', sendMessage);
    quickSearchBtn.addEventListener('click', quickSearch);

    // Upload elements
    const selectFileBtn = document.getElementById('selectFileBtn');
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');

    selectFileBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadFiles(Array.from(e.dataTransfer.files));
        }
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    chatInput.addEventListener('input', autoResizeTextarea);

    quickSearchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            quickSearch();
        }
    });

    globalSearch.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            quickSearchInput.value = globalSearch.value;
            quickSearch();
        }
    });

    // Example question buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.dataset.question;
            sendMessage();
        });
    });

    // Load uploaded documents list
    loadDocuments();
}

// Upload Functions
function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        uploadFiles(Array.from(e.target.files));
    }
}

async function uploadFiles(files) {
    if (files.length === 0) return;
    if (files.length === 1) {
        await uploadFile(files[0]);
        return;
    }

    showLoading(true, 'uploading', `Subiendo ${files.length} archivos...`);
    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < files.length; i++) {
        showLoading(true, 'uploading', `Subiendo ${i + 1}/${files.length}: ${files[i].name}`);
        const formData = new FormData();
        formData.append('file', files[i]);

        try {
            const res = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            });
            if (res.ok) {
                successCount++;
            } else {
                errorCount++;
            }
        } catch (err) {
            errorCount++;
        }
    }

    if (errorCount === 0) {
        showLoading(true, 'success', `${successCount} archivos subidos correctamente`);
    } else {
        showLoading(true, 'error', `${successCount} OK, ${errorCount} con error`);
    }

    loadStats();
    loadDocuments();
    setTimeout(() => showLoading(false), 2000);
}

async function uploadFile(file) {
    showLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (res.ok) {
            showLoading(true, 'success');
            loadStats();
            loadDocuments();
        } else {
            showLoading(true, 'error');
        }
    } catch (err) {
        showLoading(true, 'error');
    } finally {
        setTimeout(() => showLoading(false), 1500);
    }
}

async function loadDocuments() {
    const container = document.getElementById('uploadedDocs');
    try {
        const res = await fetch(`${API_BASE}/documents`);
        const data = await res.json();
        container.innerHTML = data.documents.map(doc => `
            <div class="uploaded-doc-item">
                <span class="doc-name" title="${escapeHtml(doc.titulo)}">${escapeHtml(doc.titulo)}</span>
                <button class="btn-delete" onclick="deleteDocument('${doc.id}')">Eliminar</button>
            </div>
        `).join('');
    } catch (err) {
        container.innerHTML = '';
    }
}

async function deleteDocument(docId) {
    if (!confirm('¿Eliminar este documento y todos sus chunks?')) return;

    showLoading(true);
    try {
        await fetch(`${API_BASE}/documents/${docId}`, { method: 'DELETE' });
        loadStats();
        loadDocuments();
    } catch (err) {
        alert('Error al eliminar');
    } finally {
        showLoading(false);
    }
}

// Utilities
function autoResizeTextarea() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

function showLoading(show, state = 'loading', message = null) {
    const spinner = loadingOverlay.querySelector('.spinner');
    const success = loadingOverlay.querySelector('.loading-success');
    const error = loadingOverlay.querySelector('.loading-error');
    const text = loadingOverlay.querySelector('p');

    spinner.style.display = 'none';
    success.style.display = 'none';
    error.style.display = 'none';

    if (show) {
        loadingOverlay.classList.add('active');
        if (state === 'success') {
            success.style.display = 'block';
            text.textContent = message || '';
        } else if (state === 'error') {
            error.style.display = 'block';
            text.textContent = message || '';
        } else {
            spinner.style.display = 'block';
            text.textContent = message || 'Procesando...';
        }
    } else {
        loadingOverlay.classList.remove('active');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(text, maxLength) {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

// =============================================================================
// IMAGE FUNCTIONS
// =============================================================================

const imageInput = document.getElementById('imageInput');
const selectImageBtn = document.getElementById('selectImageBtn');
const imageDropZone = document.getElementById('imageDropZone');
const imageSearchInput = document.getElementById('imageSearchInput');
const imageSearchBtn = document.getElementById('imageSearchBtn');
const imageResults = document.getElementById('imageResults');

// Image upload
selectImageBtn.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadImage(e.target.files[0]);
    }
});

// Image drag and drop
imageDropZone.addEventListener('click', () => imageInput.click());
imageDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    imageDropZone.classList.add('dragover');
});
imageDropZone.addEventListener('dragleave', () => {
    imageDropZone.classList.remove('dragover');
});
imageDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    imageDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        uploadImage(e.dataTransfer.files[0]);
    }
});

// Image search
imageSearchBtn.addEventListener('click', searchImagesByText);
imageSearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        searchImagesByText();
    }
});

async function uploadImage(file) {
    showLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/images/upload`, {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (res.ok) {
            showLoading(true, 'success');
            const counter = document.getElementById('imageCount');
            if (counter) {
                const current = parseInt(counter.textContent) || 0;
                counter.textContent = current + 1;
            }
            loadImages();
        } else {
            showLoading(true, 'error');
        }
    } catch (err) {
        showLoading(true, 'error');
    } finally {
        setTimeout(() => showLoading(false), 1500);
    }
}

async function searchImagesByText() {
    const query = imageSearchInput.value.trim();
    if (!query) return;

    imageResults.innerHTML = '<p style="color: var(--text-tertiary); font-size: 0.8rem;">Buscando...</p>';

    try {
        const res = await fetch(`${API_BASE}/images/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 }),
        });
        const data = await res.json();

        if (data.results && data.results.length > 0) {
            imageResults.innerHTML = data.results.map(r => `
                <div class="image-result-item">
                    <img src="${r.url_imagen}" alt="${escapeHtml(r.titulo)}" loading="lazy">
                    <div class="image-result-info">
                        <div class="image-result-desc">${escapeHtml(r.descripcion)}</div>
                        <div class="image-result-score">Score: ${r.score.toFixed(3)}</div>
                    </div>
                </div>
            `).join('');
        } else {
            imageResults.innerHTML = '<p style="color: var(--text-tertiary); font-size: 0.8rem;">No se encontraron imágenes</p>';
        }
    } catch (err) {
        imageResults.innerHTML = '<p style="color: var(--error); font-size: 0.8rem;">Error en la búsqueda</p>';
    }
}

function addImageResult(result) {
    const existing = imageResults.querySelector('.empty-state');
    if (existing) {
        imageResults.innerHTML = '';
    }

    const html = `
        <div class="image-result-item">
            <img src="${result.url_imagen}" alt="${escapeHtml(result.titulo)}" loading="lazy">
            <div class="image-result-info">
                <div class="image-result-desc">${escapeHtml(result.descripcion)}</div>
                ${result.etiquetas ? `
                    <div class="image-result-tags">
                        ${result.etiquetas.map(t => `<span class="tag">${t.label}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        </div>
    `;
    imageResults.insertAdjacentHTML('afterbegin', html);
}

async function loadImages() {
    try {
        const res = await fetch(`${API_BASE}/images?limit=20`);
        const data = await res.json();

        const counter = document.getElementById('imageCount');
        const total = data.total || 0;
        if (counter) counter.textContent = total;

        if (data.images && data.images.length > 0) {
            imageResults.innerHTML = data.images.map(img => `
                <div class="image-result-item">
                    <img src="${img.url_imagen}" alt="${escapeHtml(img.titulo)}" loading="lazy">
                    <div class="image-result-info">
                        <div class="image-result-desc">${escapeHtml(img.descripcion || img.titulo)}</div>
                        <button class="btn-delete" onclick="deleteImage('${img.doc_id}')" style="font-size:0.7rem;padding:2px 6px;">Eliminar</button>
                    </div>
                </div>
            `).join('');
        } else {
            imageResults.innerHTML = '<p style="color: var(--text-tertiary); font-size: 0.8rem;">No hay imágenes. Sube una para comenzar.</p>';
        }
    } catch (err) {
        imageResults.innerHTML = '';
    }
}

async function deleteImage(docId) {
    if (!confirm('¿Eliminar esta imagen?')) return;

    try {
        await fetch(`${API_BASE}/images/${docId}`, { method: 'DELETE' });
        const counter = document.getElementById('imageCount');
        if (counter) {
            const current = parseInt(counter.textContent) || 0;
            counter.textContent = Math.max(0, current - 1);
        }
        loadImages();
    } catch (err) {
        alert('Error al eliminar');
    }
}