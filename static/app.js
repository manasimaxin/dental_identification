/**
 * Dental Diagnosis AI - Smart Client Logic
 */

const state = {
  patient: null,
  stream: null,
  isAnalyzing: false,
  lastFrame: null,
  stabilityScore: 0,
  isSteady: false,
  isWellLit: false
};

// Elements
const sections = {
  reg: document.getElementById('stepRegistration'),
  camera: document.getElementById('stepCamera'),
  loading: document.getElementById('stepLoading'),
  results: document.getElementById('stepResults')
};

const video = document.getElementById('video');
const canvas = document.getElementById('stabilityCanvas');
const ctx = canvas.getContext('2d');

// Actions
document.getElementById('startCaptureBtn').addEventListener('click', startDiagnosis);
document.getElementById('backToRegBtn').addEventListener('click', () => showSection('reg'));
document.getElementById('newDiagnosisBtn').addEventListener('click', () => location.reload());
document.getElementById('closeErrorBtn').addEventListener('click', () => {
  document.getElementById('errorModal').classList.remove('active');
  startCamera();
});

function showSection(name) {
  Object.values(sections).forEach(s => s.classList.remove('active'));
  sections[name].classList.add('active');
  if (name !== 'camera' && state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
  }
}

async function startDiagnosis() {
  const name = document.getElementById('patientName').value;
  const age = document.getElementById('patientAge').value;
  const gender = document.getElementById('patientGender').value;

  if (!name || !age) {
    alert('Please enter patient name and age');
    return;
  }

  state.patient = { name, age, gender, id: Math.random().toString(36).substr(2, 9).toUpperCase() };
  showSection('camera');
  await startCamera();
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = state.stream;
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth / 4; // Downsample for performance
      canvas.height = video.videoHeight / 4;
      requestAnimationFrame(monitorFrame);
    };
  } catch (err) {
    alert('Camera access denied or not available');
    showSection('reg');
  }
}

/**
 * Monitors frames for stability and brightness
 */
function monitorFrame() {
  if (sections.camera.classList.contains('active') && !state.isAnalyzing) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // 1. Brightness check
    let totalBrightness = 0;
    for (let i = 0; i < frame.data.length; i += 4) {
      totalBrightness += (frame.data[i] + frame.data[i+1] + frame.data[i+2]) / 3;
    }
    const avgBrightness = totalBrightness / (frame.data.length / 4);
    state.isWellLit = avgBrightness > 50 && avgBrightness < 230;

    // 2. Stability check (frame differencing)
    if (state.lastFrame) {
      let diff = 0;
      for (let i = 0; i < frame.data.length; i += 4) {
        diff += Math.abs(frame.data[i] - state.lastFrame.data[i]);
      }
      const motion = diff / (frame.data.length / 4);
      
      // Accumulate stability
      if (motion < 5) {
        state.stabilityScore = Math.min(100, state.stabilityScore + 5);
      } else {
        state.stabilityScore = Math.max(0, state.stabilityScore - 10);
      }
    }
    state.lastFrame = frame;
    state.isSteady = state.stabilityScore > 80;

    updateUI();

    // Auto-capture trigger
    if (state.isSteady && state.isWellLit) {
      captureAndUpload();
      return;
    }

    requestAnimationFrame(monitorFrame);
  }
}

function updateUI() {
  const lightInd = document.getElementById('indicatorLight');
  const steadyInd = document.getElementById('indicatorSteady');
  const statusTxt = document.getElementById('cameraStatus');

  lightInd.classList.toggle('active', state.isWellLit);
  steadyInd.classList.toggle('active', state.isSteady);

  if (!state.isWellLit) {
    statusTxt.innerText = "Need more light...";
  } else if (!state.isSteady) {
    statusTxt.innerText = "Hold steady...";
  } else {
    statusTxt.innerText = "Perfect! Capturing...";
  }
}

async function captureAndUpload() {
  state.isAnalyzing = true;
  showSection('loading');

  // Capture frame
  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCanvas.getContext('2d').drawImage(video, 0, 0);
  
  captureCanvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');
    formData.append('patient_data', JSON.stringify(state.patient));

    try {
      const res = await fetch('/detect', { method: 'POST', body: formData });
      const data = await res.json();

      if (res.status === 422) {
        // Quality/Visibility Error
        showError(data.reasons);
      } else if (res.ok) {
        displayReport(data);
      } else {
        showError(['Server error. Please try again.']);
      }
    } catch (err) {
      showError(['Network error. Check connection.']);
    }
  }, 'image/jpeg', 0.9);
}

function showError(reasons) {
  state.isAnalyzing = false;
  const list = document.getElementById('errorList');
  list.innerHTML = '';
  reasons.forEach(r => {
    const li = document.createElement('li');
    li.innerText = `• ${r}`;
    list.appendChild(li);
  });
  document.getElementById('errorModal').classList.add('active');
  showSection('camera');
}

function displayReport(data) {
  showSection('results');
  document.getElementById('reportId').innerText = data.patient_id;
  
  const container = document.getElementById('reportContent');
  container.innerHTML = '';

  data.results.forEach((res, i) => {
    const item = document.createElement('div');
    item.className = 'report-item';
    
    const isUrgent = res.action_needed.includes('Urgent');
    
    item.innerHTML = `
      <img src="${data.image}" class="report-img" alt="Result Heatmap">
      <div class="type-line">${res.caries_type}</div>
      <div class="action-badge ${isUrgent ? 'urgent' : 'normal'}">${res.action_needed}</div>
      <p class="summary-text">${res.summary}</p>
      
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-val">${res.confidence}</div>
          <div class="stat-label">Confidence</div>
        </div>
        <div class="stat-item">
          <div class="stat-val">84.3%</div>
          <div class="stat-label">Model Accuracy</div>
        </div>
      </div>

      <div class="advice-box">
        <h4>Patient Advice:</h4>
        <p>${res.advice}</p>
      </div>
    `;
    container.appendChild(item);
  });
}
