const MAX_LENGTH = 5000;
const form = document.getElementById('spamForm');
const submitBtn = document.getElementById('submitBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const modelsGrid = document.getElementById('modelsGrid');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');
const gibberishAlert = document.getElementById('gibberishAlert');
const gibberishMessage = document.getElementById('gibberishMessage');
const messageInput = document.getElementById('messageInput');
const charCounter = document.getElementById('charCounter');
const sampleGrid = document.getElementById('sampleGrid');
const gibberishModal = document.getElementById('gibberishModal');
const gibberishModalMessage = document.getElementById('gibberishModalMessage');
const gibberishModalOk = document.getElementById('gibberishModalOk');

const modelConfigs = {
    logistic_regression: {
        name: 'Logistic Regression',
        icon: '📊',
        color: '#667eea'
    },
    lstm: {
        name: 'LSTM',
        icon: '🧠',
        color: '#ff6b6b'
    },
    xlm_roberta: {
        name: 'XLM-RoBERTa',
        icon: '🤖',
        color: '#51cf66'
    }
};

// Character counter
messageInput.addEventListener('input', () => {
    const length = messageInput.value.length;
    charCounter.textContent = `${length} / ${MAX_LENGTH}`;
    
    if (length > MAX_LENGTH * 0.9) {
        charCounter.classList.add('error');
        charCounter.classList.remove('warning');
    } else if (length > MAX_LENGTH * 0.7) {
        charCounter.classList.add('warning');
        charCounter.classList.remove('error');
    } else {
        charCounter.classList.remove('warning', 'error');
    }
    
    // Clear errors when typing
    messageInput.classList.remove('error-input');
    hideAlert(gibberishAlert);
    hideAlert(errorAlert);
});

// Load sample messages
async function loadSampleMessages() {
    try {
        const response = await fetch('/samples');
        const data = await response.json();
        
        // Mix spam and ham samples
        const allSamples = [
            ...data.spam.slice(0, 3).map(text => ({ text, type: 'spam' })),
            ...data.ham.slice(0, 3).map(text => ({ text, type: 'ham' }))
        ];
        
        // Shuffle
        allSamples.sort(() => Math.random() - 0.5);
        
        allSamples.forEach(sample => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'sample-btn';
            btn.innerHTML = `
                <span class="sample-badge badge-${sample.type}-sample">${sample.type}</span>
                <span class="sample-text">${sample.text}</span>
            `;
            btn.onclick = () => {
                messageInput.value = sample.text;
                messageInput.dispatchEvent(new Event('input'));
                messageInput.focus();
            };
            sampleGrid.appendChild(btn);
        });
    } catch (error) {
        console.error('Failed to load samples:', error);
    }
}

loadSampleMessages();

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = messageInput.value.trim();
    
    if (!message) {
        showAlert(errorAlert, 'Please enter a message to analyze.');
        return;
    }
    
    if (message.length > MAX_LENGTH) {
        showAlert(errorAlert, `Message is too long. Maximum ${MAX_LENGTH} characters allowed.`);
        return;
    }
    
    submitBtn.disabled = true;
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    hideAlert(errorAlert);
    hideAlert(gibberishAlert);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (response.status === 429) {
            throw new Error('Rate limit exceeded. Please wait a moment before trying again.');
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }
        
        if (data.gibberish_detected) {
            showGibberishModal(data.gibberish_reason, data.gibberish_confidence);
            loading.style.display = 'none';
            submitBtn.disabled = false;
            return;
        }

        displayResults(data, message);
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        showAlert(errorAlert, error.message);
    } finally {
        loading.style.display = 'none';
        submitBtn.disabled = false;
    }
});

function showAlert(alertElement, message) {
    const messageElement = alertElement.querySelector('span:last-child');
    messageElement.textContent = message;
    alertElement.style.display = 'block';
}

function hideAlert(alertElement) {
    alertElement.style.display = 'none';
}

function showGibberishModal(reason, confidence) {
    const reasonText = reason || 'The input contains unrecognizable or nonsensical text.';
    const confidenceText = confidence ? ` (Confidence: ${(confidence * 100).toFixed(1)}%)` : '';
    gibberishModalMessage.textContent = reasonText + confidenceText;
    gibberishModal.classList.add('show');
    messageInput.classList.add('error-input');
}

function hideGibberishModal() {
    gibberishModal.classList.remove('show');
    messageInput.classList.remove('error-input');
    messageInput.focus();
}

// Close modal handlers
gibberishModalOk.addEventListener('click', hideGibberishModal);

// Close modal when clicking outside
gibberishModal.addEventListener('click', (e) => {
    if (e.target === gibberishModal) {
        hideGibberishModal();
    }
});

// Close modal with ESC key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && gibberishModal.classList.contains('show')) {
        hideGibberishModal();
    }
});

function displayResults(data, originalMessage) {
    modelsGrid.innerHTML = '';

    ['logistic_regression', 'lstm', 'xlm_roberta'].forEach(modelKey => {
        const result = data[modelKey];
        const config = modelConfigs[modelKey];
        
        const card = createModelCard(config, result);
        modelsGrid.appendChild(card);
    });

    updateFinalVerdict(data.ensemble_verdict, originalMessage);
    displayMetricsTable(data.metadata);
}

function updateFinalVerdict(verdict, originalMessage) {
    const verdictEl = document.getElementById('finalVerdict');
    if (!verdict || !verdict.label) {
        verdictEl.textContent = '';
        return;
    }

    const safeMessage = originalMessage || '';
    const label = verdict.label.toUpperCase();
    if (label === 'SPAM') {
        verdictEl.innerHTML = `The message: "${safeMessage}" <br> is <span class="spam-label">SPAM</span> (${verdict.spam_votes}/3 models agreed)`;
    } else if (label === 'HAM') {
        verdictEl.innerHTML = `The message: "${safeMessage}" <br> is <span class="ham-label">HAM</span> (${verdict.ham_votes}/3 models agreed)`;
    } else {
        verdictEl.textContent = `The message: "${safeMessage}" <br> is Inconclusive (no 2-out-of-3 majority)`;
    }
}

function createModelCard(config, result) {
    const card = document.createElement('div');
    card.className = 'model-card';

    if (result.error) {
        card.innerHTML = `
            <div class="model-header">
                <div class="model-icon">${config.icon}</div>
                <div class="model-name">${config.name}</div>
            </div>
            <div style="padding: 20px; background: rgba(255, 69, 58, 0.1); border-radius: 12px; color: #ff453a;">
                ⚠️ ${result.error}
            </div>
        `;
        return card;
    }

    const isSpam = result.label === 'SPAM';
    const badgeClass = isSpam ? 'badge-spam' : 'badge-ham';
    const fillClass = isSpam ? 'fill-spam' : 'fill-ham';

    card.innerHTML = `
        <div class="model-header">
            <div class="model-icon">${config.icon}</div>
            <div class="model-name">${config.name}</div>
        </div>
        <div>
            <span class="prediction-badge ${badgeClass}">
                ${isSpam ? '🚫' : '✅'} ${result.label}
            </span>
            <div class="confidence-bar">
                <div class="confidence-fill ${fillClass}" style="width: ${result.confidence * 100}%"></div>
            </div>
            <div style="text-align: center; margin: 12px 0;">
                <span style="font-size: 24px; font-weight: 600; color: #f5f5f7;">
                    ${(result.confidence * 100).toFixed(1)}%
                </span>
                <span style="font-size: 13px; color: #86868b; margin-left: 4px;">confident</span>
            </div>
            <div class="probabilities">
                <div class="prob-item">
                    <span>🚫 Spam</span>
                    <span class="prob-value">${(result.spam_probability * 100).toFixed(2)}%</span>
                </div>
                <div class="prob-item" style="text-align: right;">
                    <span>✅ Ham</span>
                    <span class="prob-value">${(result.ham_probability * 100).toFixed(2)}%</span>
                </div>
            </div>
        </div>
    `;

    return card;
}

function displayMetricsTable(metadata) {
    const tbody = document.getElementById('metricsTableBody');
    tbody.innerHTML = '';

    const models = [
        { key: 'logistic_regression', icon: '📊' },
        { key: 'lstm', icon: '🧠' },
        { key: 'xlm_roberta', icon: '🤖' }
    ];

    models.forEach(model => {
        const meta = metadata[model.key];
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 24px;">${model.icon}</span>
                    <div>
                        <div style="font-weight: 600;">${meta.name}</div>
                        <div class="model-description">${meta.description}</div>
                    </div>
                </div>
            </td>
            <td><strong>${(meta.accuracy * 100).toFixed(2)}%</strong></td>
            <td>${(meta.precision * 100).toFixed(2)}%</td>
            <td>${(meta.recall * 100).toFixed(2)}%</td>
            <td>${(meta.f1 * 100).toFixed(2)}%</td>
            <td><span class="metric-badge">${meta.training_time}</span></td>
        `;
        
        tbody.appendChild(row);
    });
}
