const canvas = document.getElementById('whiteboard');
const ctx = canvas.getContext('2d');
const outputEl = document.getElementById('output');
const clearBtn = document.getElementById('clear-btn');

// --- Application State ---
let session = null;
let isDrawing = false;
let debounceTimer = null;
let isPredicting = false; // Mutex lock to prevent session collisions

const LABELS = ['cloud', 'database', 'user', 'server', 'arrow', 'rectangle'];

// --- Canvas Configuration ---
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000000';

// Helper to get exact mouse coordinates relative to the canvas
function getMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top
  };
}

// --- 1. Initialize ONNX Engine ---
async function initAI() {
  try {
    outputEl.innerHTML = 'Status: <span class="highlight">Loading Model...</span>';
    
    // Load the single, self-contained model
    session = await ort.InferenceSession.create('/static/model/sketch_classifier_single.onnx', {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all'
    });
    
    outputEl.innerHTML = 'Status: <span class="highlight">Ready (WebGPU Active)</span>';
  } catch (err) {
    console.error('Inference Engine Error:', err);
    outputEl.innerHTML = 'Status: <span style="color: red">Failed to load AI engine</span>';
  }
}

// --- 2. Convert Canvas to Tensor ---
function getCanvasTensor() {
  const targetSize = 28;
  const offscreen = document.createElement('canvas');
  offscreen.width = targetSize;
  offscreen.height = targetSize;
  const oCtx = offscreen.getContext('2d');
  
  // Scale down drawing to 28x28
  oCtx.drawImage(canvas, 0, 0, targetSize, targetSize);
  const imgData = oCtx.getImageData(0, 0, targetSize, targetSize).data;
  const input = new Float32Array(targetSize * targetSize);

  for (let i = 0; i < input.length; i++) {
    // Extract Alpha channel and normalize: 0.0 (white) to 1.0 (black ink)
    const alpha = imgData[i * 4 + 3];
    input[i] = alpha / 255.0;
  }

  // Reshape to [Batch=1, Channels=1, Height=28, Width=28]
  return new ort.Tensor('float32', input, [1, 1, targetSize, targetSize]);
}

// --- 3. Run Inference ---
async function predict() {
  if (!session) return;
  
  // Drop the frame if the engine is already processing a request
  if (isPredicting) return; 
  isPredicting = true; // Lock

  try {
    const t0 = performance.now();
    const tensor = getCanvasTensor();
    const feeds = { [session.inputNames[0]]: tensor };
    
    const results = await session.run(feeds);
    
    const output = results[session.outputNames[0]].data;
    let maxIdx = 0;
    let maxVal = -Infinity;

    for (let i = 0; i < output.length; i++) {
      if (output[i] > maxVal) {
        maxVal = output[i];
        maxIdx = i;
      }
    }

    const elapsed = (performance.now() - t0).toFixed(1);
    const label = LABELS[maxIdx] || 'Unknown';
    outputEl.innerHTML = `Detected: <span class="highlight">${label}</span> (${elapsed} ms)`;
  } catch (err) {
    console.error("Prediction failed:", err);
  } finally {
    // Always unlock when finished or on error
    isPredicting = false; 
  }
}

// --- 4. Canvas Drawing Listeners ---
canvas.addEventListener('pointerdown', (e) => {
  isDrawing = true;
  const pos = getMousePos(e);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
});

canvas.addEventListener('pointermove', (e) => {
  if (!isDrawing) return;
  const pos = getMousePos(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();

  // Debounce real-time prediction so it runs 250ms after the mouse pauses
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(predict, 250);
});

window.addEventListener('pointerup', () => {
  if (isDrawing) {
    isDrawing = false;
    // Clear debounce to prevent simultaneous predictions on mouse up
    clearTimeout(debounceTimer); 
    predict();
  }
});

clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  outputEl.innerHTML = 'Status: <span class="highlight">Canvas Cleared</span>';
});

// Boot the AI
initAI();