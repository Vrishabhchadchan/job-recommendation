/* =====================================================
   SMART JOB MATCH — Application Logic
   ===================================================== */

// ── Config ──────────────────────────────────────────
const API_BASE = '';  // same-origin; works locally and on Vercel
// Replace with your Google Cloud OAuth 2.0 Client ID to enable real sign-in.
// Leave as empty string to use the built-in demo sign-in.
const GOOGLE_CLIENT_ID = '240437531130-1gr27utv5pjrhe0uh2i7f0amsl6j36jb.apps.googleusercontent.com';

// Set PDF.js worker
if (typeof pdfjsLib !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

// ── State ────────────────────────────────────────────
let currentUser      = null;
let currentResume    = '';   // extracted text
let currentFileName  = '';
let lastAPIResponse  = null; // original /recommend response (never overwritten)
let pendingQuestion  = '';   // question currently shown to the user in the CQ card

// ── DOM shortcuts ────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Google Auth ──────────────────────────────────────

// Restore session from localStorage on every page load
window.addEventListener('load', () => {
  const saved = localStorage.getItem('sjm_user');
  if (saved) {
    try { setUser(JSON.parse(saved), /* silent */ true); } catch { localStorage.removeItem('sjm_user'); }
  }
});

function triggerGoogleSignIn() {
  if (GOOGLE_CLIENT_ID && typeof google !== 'undefined' && google.accounts?.oauth2) {
    // Use the OAuth2 token popup — always opens a sign-in window on button click
    const tokenClient = google.accounts.oauth2.initTokenClient({
      client_id: GOOGLE_CLIENT_ID,
      scope: 'https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email',
      callback: async (resp) => {
        if (resp.error) {
          showToast('Sign-in cancelled or failed.', 'error');
          return;
        }
        try {
          const r = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
            headers: { Authorization: `Bearer ${resp.access_token}` },
          });
          const info = await r.json();
          setUser({ name: info.name, email: info.email, avatar: info.picture });
        } catch {
          showToast('Could not fetch Google profile.', 'error');
        }
      },
    });
    tokenClient.requestAccessToken({ prompt: '' });
  } else {
    demoSignIn();
  }
}

function demoSignIn() {
  const btn = $('google-btn');
  btn.disabled = true;
  btn.textContent = 'Signing in...';
  setTimeout(() => {
    setUser({
      name:   'Demo User',
      email:  'demo@example.com',
      avatar: `https://ui-avatars.com/api/?name=Demo+User&background=6366f1&color=fff&size=64`,
    });
  }, 900);
}

function setUser(user, silent = false) {
  currentUser = user;
  localStorage.setItem('sjm_user', JSON.stringify(user));
  $('user-avatar').src = user.avatar;
  $('user-name').textContent = user.name;
  $('signin-wrap').classList.add('hidden');
  $('user-wrap').classList.remove('hidden');
  if (!silent) showToast(`Welcome, ${user.name.split(' ')[0]}!`, 'success');
}

function signOut() {
  currentUser = null;
  localStorage.removeItem('sjm_user');
  $('signin-wrap').classList.remove('hidden');
  $('user-wrap').classList.add('hidden');
  const btn = $('google-btn');
  btn.disabled = false;
  btn.innerHTML = `<svg class="g-icon" viewBox="0 0 24 24">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
  </svg> Sign in with Google`;
}

function parseJWT(token) {
  try {
    return JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
  } catch { return {}; }
}

// ── Drag & Drop ──────────────────────────────────────
function onDragEnter(e) {
  e.preventDefault();
  setDropState('hover');
}

function onDragOver(e) {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
  $('drop-zone').classList.add('drag-over');
}

function onDragLeave(e) {
  // Only react if leaving the zone entirely
  if (!$('drop-zone').contains(e.relatedTarget)) {
    setDropState('idle');
    $('drop-zone').classList.remove('drag-over');
  }
}

function onDrop(e) {
  e.preventDefault();
  $('drop-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
}

function triggerFilePicker() {
  // Don't open picker if a file is already selected (let the chip handle removal)
  if (currentFileName) return;
  $('file-input').click();
}

function onFileChange(e) {
  const file = e.target.files[0];
  if (file) handleFile(file);
}

function removeFile(e) {
  e.stopPropagation();
  currentResume   = '';
  currentFileName = '';
  $('file-input').value = '';
  setDropState('idle');
  updateAnalyzeBtn();
}

function setDropState(state) {
  $('dz-idle').classList.toggle('hidden',     state !== 'idle');
  $('dz-hover').classList.toggle('hidden',    state !== 'hover');
  $('dz-selected').classList.toggle('hidden', state !== 'selected');
}

// ── File Processing ───────────────────────────────────
const ACCEPTED = ['application/pdf', 'text/plain',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];

async function handleFile(file) {
  const type = file.type;
  if (!ACCEPTED.includes(type) && !file.name.match(/\.(pdf|txt|doc|docx)$/i)) {
    showToast('Unsupported file type. Please upload PDF, TXT, or DOCX.', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File is too large. Maximum size is 10 MB.', 'error');
    return;
  }

  currentFileName = file.name;
  $('file-chip-name').textContent = file.name;
  setDropState('selected');

  try {
    if (file.name.endsWith('.pdf') || type === 'application/pdf') {
      currentResume = await extractPDF(file);
    } else if (file.name.endsWith('.docx') || type.includes('wordprocessingml')) {
      currentResume = await extractDOCX(file);
    } else {
      currentResume = await extractTXT(file);
    }

    // Sync into textarea so the user can see and edit
    $('resume-ta').value = currentResume;
    $('char-count').textContent = `${currentResume.length} characters`;
    updateAnalyzeBtn();
    showToast('Resume extracted successfully!', 'success');
  } catch (err) {
    showToast('Could not extract text from file. Try pasting manually.', 'error');
    console.error(err);
    currentResume   = '';
    currentFileName = '';
    setDropState('idle');
    updateAnalyzeBtn();
  }
}

function extractTXT(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = e => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsText(file, 'UTF-8');
  });
}

async function extractPDF(file) {
  const buffer = await file.arrayBuffer();
  const pdf    = await pdfjsLib.getDocument({ data: buffer }).promise;
  const pages  = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page    = await pdf.getPage(i);
    const content = await page.getTextContent();
    pages.push(content.items.map(it => it.str).join(' '));
  }
  return pages.join('\n');
}

async function extractDOCX(file) {
  const buffer = await file.arrayBuffer();
  const result = await mammoth.extractRawText({ arrayBuffer: buffer });
  return result.value;
}

// ── Textarea ──────────────────────────────────────────
function onTextInput() {
  const val = $('resume-ta').value;
  currentResume = val;
  $('char-count').textContent = `${val.length} characters`;
  // If user types, clear the file selection to avoid confusion
  if (val && currentFileName) {
    currentFileName = '';
    setDropState('idle');
  }
  updateAnalyzeBtn();
}

function updateAnalyzeBtn() {
  $('analyze-btn').disabled = !currentResume.trim();
}

// ── Analysis ──────────────────────────────────────────
async function runAnalysis() {
  const text = currentResume.trim();
  if (!text) return;

  showLoading();
  const stepTimer = animateLoadingSteps();

  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resume_text: text }),
    });

    clearTimeout(stepTimer);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    lastAPIResponse = await res.json();
    hideLoading();
    renderResults(lastAPIResponse);
    showView('results');
  } catch (err) {
    hideLoading();
    showToast(`Analysis failed: ${err.message}`, 'error');
  }
}

// ── Loading ───────────────────────────────────────────
function showLoading() {
  $('loading-screen').classList.remove('hidden');
  // Reset progress steps
  ['prog-1','prog-2','prog-3'].forEach(id => {
    const el = $(id);
    el.classList.remove('active','done');
    el.querySelector('.prog-dot').style.background = '';
    el.querySelector('.prog-check').classList.add('hidden');
  });
  $('prog-1').classList.add('active');
}

function hideLoading() {
  $('loading-screen').classList.add('hidden');
}

function animateLoadingSteps() {
  let step = 1;
  const advance = () => {
    if (step >= 3) return;
    const cur = $(`prog-${step}`);
    cur.classList.remove('active');
    cur.classList.add('done');
    cur.querySelector('.prog-check').classList.remove('hidden');
    step++;
    const next = $(`prog-${step}`);
    next.classList.add('active');
  };
  const t1 = setTimeout(advance, 3500);
  const t2 = setTimeout(advance, 8000);
  return t2; // return last timer id (caller only clears one, that's fine)
}

// ── Results Rendering (Claude-like) ───────────────────
function renderResults(data) {
  const container = $('claude-response');
  container.innerHTML = '';

  // 1. Document chip
  container.appendChild(buildDocChip());

  // 2. AI message
  const msg = document.createElement('div');
  msg.className = 'ai-message';
  msg.innerHTML = buildAIMessage(data);
  container.appendChild(msg);

  // After DOM insert, animate match bars
  requestAnimationFrame(() => {
    container.querySelectorAll('.match-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });
}

function buildDocChip() {
  const name = currentFileName || 'Pasted Resume';
  const div  = document.createElement('div');
  div.className = 'doc-chip';
  div.innerHTML = `
    <div class="doc-chip-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
      </svg>
    </div>
    <span class="doc-chip-name">${escHtml(name)}</span>
    <div class="doc-chip-status">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5"/>
      </svg>
    </div>`;
  return div;
}

function buildAIMessage(data) {
  const c = data.candidate;
  const firstName = c.name.split(' ')[0] || 'there';

  return `
    <p class="ai-greeting">
      Hi <strong>${escHtml(firstName)}</strong>! I've analyzed your resume and found your top career matches.
      Here's a full breakdown of your profile and the best-fit roles across our job database.
    </p>

    <div class="ai-section-title">Your Profile</div>
    ${buildProfileCard(c)}

    <div class="ai-section-title">${data.ranked_jobs.length} Best Matches</div>
    <div class="jobs-list">
      ${data.ranked_jobs.map((job, i) => buildJobCard(job, i)).join('')}
    </div>

    <div class="ai-section-title">One Quick Question</div>
    ${buildCQCard(data.clarifying_question)}
  `;
}

function buildProfileCard(c) {
  const skills = (c.skills || []).slice(0, 10);
  const roles  = (c.preferred_roles || []).join(', ') || 'Not specified';
  const exp    = c.experience_years
    ? `${c.experience_years} year${c.experience_years !== 1 ? 's' : ''}`
    : 'Not specified';

  return `
    <div class="profile-card">
      <div class="profile-row">
        <div class="profile-field">
          <div class="profile-label">Name</div>
          <div class="profile-value">${escHtml(c.name || 'Unknown')}</div>
        </div>
        <div class="profile-field">
          <div class="profile-label">Experience</div>
          <div class="profile-value">${escHtml(exp)}</div>
        </div>
        <div class="profile-field">
          <div class="profile-label">Education</div>
          <div class="profile-value">${escHtml(c.education || 'Not specified')}</div>
        </div>
        <div class="profile-field" style="flex-basis:100%">
          <div class="profile-label">Preferred Roles</div>
          <div class="profile-value">${escHtml(roles)}</div>
        </div>
        ${skills.length ? `
        <div class="profile-field" style="flex-basis:100%">
          <div class="profile-label">Skills</div>
          <div class="skill-pills">
            ${skills.map(s => `<span class="skill-pill">${escHtml(s)}</span>`).join('')}
          </div>
        </div>` : ''}
      </div>
    </div>`;
}

function buildJobCard(job, index) {
  const pct   = Math.round((job.similarity_score || 0) * 100);
  const level = matchLevel(job.similarity_score);
  const color = companyColor(job.company);
  const init  = companyInitials(job.company);

  const locationHTML = job.location
    ? `<span class="job-location">📍 ${escHtml(job.location)}</span>` : '';
  const remoteHTML = job.is_remote
    ? `<span class="remote-badge">🌐 Remote</span>` : '';
  const applyHTML = job.apply_link
    ? `<a class="apply-btn" href="${escHtml(job.apply_link)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">
         Apply Now
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" width="14" height="14">
           <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25"/>
         </svg>
       </a>` : '';

  return `
    <div class="job-card">
      <div class="job-header">
        <div class="job-avatar" style="background:${color}">${escHtml(init)}</div>
        <div class="job-meta">
          <div class="job-rank-title">
            <span class="job-rank">#${index + 1}</span>
            <span class="job-title">${escHtml(job.title)}</span>
          </div>
          <div class="job-company-row">
            <span class="job-company">${escHtml(job.company)}</span>
            ${locationHTML}
            ${remoteHTML}
          </div>
        </div>
        <div class="match-badge match-${level.cls}">
          <span class="match-pct">${pct}%</span>
          <span class="match-label">${level.label}</span>
        </div>
      </div>
      <div class="match-bar-wrap">
        <div class="match-bar-track">
          <div class="match-bar-fill" style="width:0%" data-width="${pct}%"></div>
        </div>
      </div>
      <p class="job-explanation">"${escHtml(job.explanation)}"</p>
      ${applyHTML ? `<div class="job-footer">${applyHTML}</div>` : ''}
    </div>`;
}

function buildCQCard(question, icon = '💬', title = 'Refine your results', placeholder = 'Type your answer...') {
  pendingQuestion = question;
  return `
    <div class="cq-card">
      <div class="cq-header">
        <span class="cq-icon">${icon}</span>
        <span class="cq-title">${title}</span>
      </div>
      <p class="cq-text">${escHtml(question)}</p>
      <div class="cq-input-row">
        <input id="cq-input" class="cq-input" type="text"
               placeholder="${placeholder}"
               onkeypress="if(event.key==='Enter') refineResults()" />
        <button class="refine-btn" onclick="refineResults()">
          Refine
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3"/>
          </svg>
        </button>
      </div>
    </div>`;
}

// ── Refine ────────────────────────────────────────────
async function refineResults() {
  const answer = $('cq-input')?.value?.trim();
  if (!answer) { showToast('Please type an answer first.'); return; }
  if (!lastAPIResponse) return;

  showLoading();

  try {
    const res = await fetch(`${API_BASE}/refine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        resume_text:         currentResume,
        clarifying_question: pendingQuestion,   // always the currently-shown question
        candidate_answer:    answer,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const refined = await res.json();
    hideLoading();
    renderRefinedResults(refined, answer);
  } catch (err) {
    hideLoading();
    showToast(`Refinement failed: ${err.message}`, 'error');
  }
}

function renderRefinedResults(data, answer) {
  const container = $('claude-response');
  container.innerHTML = '';

  // Doc chip
  container.appendChild(buildDocChip());

  // ← Back to original results button
  const backBtn = document.createElement('button');
  backBtn.className = 'back-original-btn';
  backBtn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" width="15" height="15">
      <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18"/>
    </svg>
    Back to original results`;
  backBtn.onclick = () => renderResults(lastAPIResponse);
  container.appendChild(backBtn);

  // Reasoning card
  const rc = document.createElement('div');
  rc.className = 'reasoning-card';
  rc.innerHTML = `<strong>Updated results</strong> based on your answer: "${escHtml(answer)}"<br/><br/>${escHtml(data.reasoning)}`;
  container.appendChild(rc);

  // Updated jobs + refine-again section
  const firstName = lastAPIResponse?.candidate?.name?.split(' ')[0] || 'there';
  const msg = document.createElement('div');
  msg.className = 'ai-message';
  msg.innerHTML = `
    <p class="ai-greeting">Here are your refined job matches, ${escHtml(firstName)}.</p>
    <div class="ai-section-title">${data.ranked_jobs.length} Refined Matches</div>
    <div class="jobs-list">
      ${data.ranked_jobs.map((job, i) => buildJobCard(job, i)).join('')}
    </div>
    <div class="ai-section-title">Refine Again</div>
    ${buildCQCard(
      'Want to narrow down further? Add any extra preferences — location, industry, work style, salary range, or anything else.',
      '🔄',
      'Refine again',
      'e.g. remote only, fintech, senior roles, New York...'
    )}`;
  container.appendChild(msg);

  requestAnimationFrame(() => {
    container.querySelectorAll('.match-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });
}

// ── View Navigation ───────────────────────────────────
function showView(name) {
  $('view-upload').classList.toggle('active',  name === 'upload');
  $('view-upload').classList.toggle('hidden',  name !== 'upload');
  $('view-results').classList.toggle('active', name === 'results');
  $('view-results').classList.toggle('hidden', name !== 'results');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function goBack() {
  showView('upload');
  // Reset file state
  currentResume   = '';
  currentFileName = '';
  $('file-input').value = '';
  $('resume-ta').value  = '';
  $('char-count').textContent = '0 characters';
  setDropState('idle');
  updateAnalyzeBtn();
  lastAPIResponse = null;
}

// ── Helpers ───────────────────────────────────────────
function matchLevel(score) {
  if (score >= 0.70) return { cls: 'excellent', label: 'Excellent' };
  if (score >= 0.55) return { cls: 'strong',    label: 'Strong' };
  if (score >= 0.40) return { cls: 'good',      label: 'Good' };
  return               { cls: 'low',       label: 'Potential' };
}

function companyColor(name) {
  const palette = ['#6366f1','#8b5cf6','#14b8a6','#f59e0b','#ec4899','#3b82f6','#10b981','#f97316'];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = ((hash << 5) - hash) + name.charCodeAt(i);
  return palette[Math.abs(hash) % palette.length];
}

function companyInitials(name) {
  return name.split(/\s+/).slice(0, 2).map(w => w[0] || '').join('').toUpperCase();
}

function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ── Toast ─────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}
