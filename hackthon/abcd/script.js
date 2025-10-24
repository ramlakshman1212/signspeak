class AlphabetDisplayApp {
    constructor() {
        this.words = [];
        this.currentWord = '';
        this.typingTimeout = null;
        this.initializeElements();
        this.bindEvents();
        this.updateStats();
        this.updateTime();
        setInterval(() => this.updateTime(), 1000);
    }

    initializeElements() {
        this.textInput = document.getElementById('textInput');
        this.showBtn = document.getElementById('showBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.wordsContainer = document.getElementById('wordsContainer');
        this.statusText = document.getElementById('statusText');
        this.wordCount = document.getElementById('wordCount');
        this.letterCount = document.getElementById('letterCount');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.wordModal = document.getElementById('wordModal');
        this.modalTitle = document.getElementById('modalTitle');
        this.modalImages = document.getElementById('modalImages');
        this.closeModal = document.getElementById('closeModal');
        this.exportBtn = document.getElementById('exportBtn');
        this.totalWords = document.getElementById('totalWords');
        this.totalImages = document.getElementById('totalImages');
        this.currentTime = document.getElementById('currentTime');
        
        // ASL Learning elements
        this.aslBtn = document.getElementById('aslBtn');
        this.aslModal = document.getElementById('aslModal');
        this.closeAslModal = document.getElementById('closeAslModal');
        this.aslInput = document.getElementById('aslInput');
        this.learnAslBtn = document.getElementById('learnAslBtn');
        this.aslResult = document.getElementById('aslResult');
        this.prevAslBtn = document.getElementById('prevAslBtn');
        this.nextAslBtn = document.getElementById('nextAslBtn');
        this.autoAslBtn = document.getElementById('autoAslBtn');
        
        // Voice recognition setup
        this.recognition = null;
        this.isListening = false;
        this.setupVoiceRecognition();
        
        // Gemini API setup
        this.geminiApiKey = "AIzaSyB4gYz5TbwbrmzWAF8ddQ8A55Dt4x5Pkc4";
        
        // ASL lesson progression
        this.currentAslLetter = 0;
        this.aslAlphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        this.autoAslInterval = null;
        this.isAutoAsl = false;
    }

    bindEvents() {
        // Real-time text input with automatic image display
        this.textInput.addEventListener('input', (e) => {
            this.currentWord = e.target.value;
            
            // Clear any existing timeout
            clearTimeout(this.typingTimeout);
            
            if (this.currentWord.trim()) {
                this.updateStatus(`⏳ Typing: "${this.currentWord}" - Will auto-display in 1 second...`);
                
                // Auto-add word when user stops typing (debounced)
                this.typingTimeout = setTimeout(() => {
                    if (this.currentWord.trim()) {
                        this.addWord();
                    }
                }, 1000); // Wait 1 second after user stops typing
            } else {
                this.updateStatus('✨ Ready - Type some text to see alphabet images');
            }
        });

        // Show button
        this.showBtn.addEventListener('click', () => {
            this.addWord();
        });

        // Clear button
        this.clearBtn.addEventListener('click', () => {
            this.clearAll();
        });

        // Keyboard shortcuts
        this.textInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.addWord();
            } else if (e.key === 'Escape') {
                this.clearAll();
            }
        });

        // Modal events
        this.closeModal.addEventListener('click', () => {
            this.closeWordModal();
        });

        this.wordModal.addEventListener('click', (e) => {
            if (e.target === this.wordModal) {
                this.closeWordModal();
            }
        });

        // Voice button
        this.voiceBtn.addEventListener('click', () => {
            this.toggleVoiceRecognition();
        });

        // ASL Learning button
        this.aslBtn.addEventListener('click', () => {
            this.openAslModal();
        });

        // ASL Modal events
        this.closeAslModal.addEventListener('click', () => {
            this.closeAslModalHandler();
        });

        this.aslModal.addEventListener('click', (e) => {
            if (e.target === this.aslModal) {
                this.closeAslModalHandler();
            }
        });

        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.aslModal.style.display === 'block') {
                this.closeAslModalHandler();
            }
        });

        // ASL Learning button
        this.learnAslBtn.addEventListener('click', () => {
            this.learnAslSign();
        });

        // ASL Input enter key
        this.aslInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.learnAslSign();
            }
        });
        
        // ASL Navigation buttons
        this.prevAslBtn.addEventListener('click', () => {
            this.previousAslLetter();
        });
        
        this.nextAslBtn.addEventListener('click', () => {
            this.nextAslLetter();
        });
        
        this.autoAslBtn.addEventListener('click', () => {
            this.toggleAutoAsl();
        });

        // Export button
        this.exportBtn.addEventListener('click', () => {
            this.exportWords();
        });

        // Focus on input
        this.textInput.focus();
    }

    extractLetters(text) {
        // Extract only alphabetic characters and convert to uppercase
        return text.replace(/[^a-zA-Z]/g, '').toUpperCase().split('');
    }

    extractLettersWithSpaces(text) {
        // Extract letters and preserve word boundaries with space indicators
        const words = text.trim().split(/\s+/);
        const result = [];
        
        words.forEach((word, wordIndex) => {
            const letters = word.replace(/[^a-zA-Z]/g, '').toUpperCase().split('');
            result.push(...letters);
            
            // Add space indicator between words (except after the last word)
            if (wordIndex < words.length - 1) {
                result.push('SPACE');
            }
        });
        
        return result;
    }

    async addWord() {
        const text = this.textInput.value.trim();
        if (!text) {
            this.updateStatus('⚠️ Please enter some text first', 'warning');
            return;
        }

        const letters = this.extractLettersWithSpaces(text);
        if (letters.length === 0) {
            this.updateStatus('⚠️ No valid letters found in input', 'warning');
            return;
        }

        this.showLoading();

        try {
            const wordData = {
                id: Date.now(),
                text: text,
                letters: letters,
                timestamp: new Date(),
                images: await this.loadImagesWithSpaces(letters)
            };

            this.words.unshift(wordData); // Add to beginning of array
            this.renderWords();
            this.updateStats();
            this.textInput.value = '';
            this.currentWord = '';
            this.updateStatus(`✅ Added "${text}" with ${letters.length} letters`, 'success');
        } catch (error) {
            console.error('Error adding word:', error);
            this.updateStatus('❌ Error loading images', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadImages(letters) {
        const imagePromises = letters.map(async (letter) => {
            try {
                const response = await fetch(`alphabets/${letter}.jpg`);
                if (response.ok) {
                    return {
                        letter: letter,
                        url: `alphabets/${letter}.jpg`,
                        exists: true
                    };
                } else {
                    throw new Error('Image not found');
                }
            } catch (error) {
                return {
                    letter: letter,
                    url: null,
                    exists: false
                };
            }
        });

        return Promise.all(imagePromises);
    }

    async loadImagesWithSpaces(letters) {
        const imagePromises = letters.map(async (letter) => {
            if (letter === 'SPACE') {
                return {
                    letter: 'SPACE',
                    url: null,
                    exists: true,
                    isSpace: true
                };
            }
            
            try {
                const response = await fetch(`alphabets/${letter}.jpg`);
                if (response.ok) {
                    return {
                        letter: letter,
                        url: `alphabets/${letter}.jpg`,
                        exists: true,
                        isSpace: false
                    };
                } else {
                    throw new Error('Image not found');
                }
            } catch (error) {
                return {
                    letter: letter,
                    url: null,
                    exists: false,
                    isSpace: false
                };
            }
        });

        return Promise.all(imagePromises);
    }

    renderWords() {
        if (this.words.length === 0) {
            this.wordsContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">
                        <i class="fas fa-magic"></i>
                    </div>
                    <h3>Ready to Create Magic?</h3>
                    <p>Type any word and watch it transform into beautiful alphabet images!</p>
                    <div class="empty-features">
                        <div class="feature">
                            <i class="fas fa-bolt"></i>
                            <span>Auto-display</span>
                        </div>
                        <div class="feature">
                            <i class="fas fa-history"></i>
                            <span>History preserved</span>
                        </div>
                        <div class="feature">
                            <i class="fas fa-expand"></i>
                            <span>Click to enlarge</span>
                        </div>
                    </div>
                </div>
            `;
            return;
        }

        // Always show ALL words - never disable previous ones
        this.wordsContainer.innerHTML = this.words.map((word, index) => this.createWordCard(word, index)).join('');
        
        // Add click events to word cards
        this.wordsContainer.querySelectorAll('.word-card').forEach((card, index) => {
            card.addEventListener('click', () => {
                this.openWordModal(this.words[index]);
            });
        });
    }

    createWordCard(word, index) {
        const timeAgo = this.getTimeAgo(word.timestamp);
        const imagesHtml = word.images.map(img => {
            if (img.isSpace) {
                return `
                    <div class="space-indicator">
                        <div class="space-line"></div>
                        <div class="space-label">SPACE</div>
                    </div>
                `;
            } else if (img.exists) {
                return `
                    <div class="letter-image">
                        <img src="${img.url}" alt="${img.letter}" loading="lazy">
                        <div class="letter-label">${img.letter}</div>
                    </div>
                `;
            } else {
                return `
                    <div class="letter-placeholder">
                        <div>${img.letter}?</div>
                        <div style="font-size: 0.6rem;">Missing</div>
                    </div>
                `;
            }
        }).join('');

        return `
            <div class="word-card fade-in" data-word-index="${index}">
                <div class="word-header">
                    <div class="word-text">${word.text}</div>
                    <div class="word-meta">
                        <span><i class="fas fa-clock"></i> ${timeAgo}</span>
                        <span><i class="fas fa-font"></i> ${word.letters.length} letters</span>
                        <span><i class="fas fa-image"></i> ${word.images.filter(img => img.exists).length} images</span>
                    </div>
                </div>
                <div class="word-images">
                    ${imagesHtml}
                </div>
            </div>
        `;
    }

    openWordModal(word) {
        this.modalTitle.textContent = `"${word.text}" - ${word.letters.length} letters`;
        
        this.modalImages.innerHTML = word.images.map(img => {
            if (img.exists) {
                return `
                    <div class="letter-image">
                        <img src="${img.url}" alt="${img.letter}">
                        <div class="letter-label">${img.letter}</div>
                    </div>
                `;
            } else {
                return `
                    <div class="letter-placeholder">
                        <div>${img.letter}?</div>
                        <div style="font-size: 0.6rem;">Missing</div>
                    </div>
                `;
            }
        }).join('');

        this.wordModal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    closeWordModal() {
        this.wordModal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }

    clearAll() {
        this.words = [];
        this.textInput.value = '';
        this.currentWord = '';
        this.renderWords();
        this.updateStats();
        this.updateStatus('🗑️ All words cleared', 'info');
    }

    updateStats() {
        const totalLetters = this.words.reduce((sum, word) => sum + word.letters.length, 0);
        const totalImages = this.words.reduce((sum, word) => sum + word.images.filter(img => img.exists).length, 0);
        
        this.totalWords.textContent = this.words.length;
        this.totalImages.textContent = totalImages;
        
        if (this.wordCount) this.wordCount.textContent = `${this.words.length} word${this.words.length !== 1 ? 's' : ''}`;
        if (this.letterCount) this.letterCount.textContent = `${totalLetters} letter${totalLetters !== 1 ? 's' : ''}`;
    }

    updateTime() {
        if (this.currentTime) {
            const now = new Date();
            this.currentTime.textContent = now.toLocaleTimeString();
        }
    }

    setupVoiceRecognition() {
        // Check if browser supports speech recognition
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.voiceBtn.style.display = 'none';
            console.log('Speech recognition not supported in this browser');
            return;
        }

        // Initialize speech recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isListening = true;
            this.voiceBtn.classList.add('listening');
            this.voiceBtn.innerHTML = '<i class="fas fa-microphone-slash"></i> Stop Listening';
            this.updateStatus('🎤 Listening... Speak now!', 'info');
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.textInput.value = transcript;
            this.currentWord = transcript;
            this.updateStatus(`🎤 Voice input: "${transcript}"`, 'success');
            
            // Auto-trigger word addition after voice input
            setTimeout(() => {
                this.addWord();
            }, 500);
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.updateStatus(`❌ Voice error: ${event.error}`, 'error');
            this.stopVoiceRecognition();
        };

        this.recognition.onend = () => {
            this.stopVoiceRecognition();
        };
    }

    toggleVoiceRecognition() {
        if (this.isListening) {
            this.stopVoiceRecognition();
        } else {
            this.startVoiceRecognition();
        }
    }

    startVoiceRecognition() {
        if (!this.recognition) {
            this.updateStatus('❌ Voice recognition not supported', 'error');
            return;
        }

        try {
            this.recognition.start();
        } catch (error) {
            console.error('Error starting voice recognition:', error);
            this.updateStatus('❌ Error starting voice recognition', 'error');
        }
    }

    stopVoiceRecognition() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
        
        this.isListening = false;
        this.voiceBtn.classList.remove('listening');
        this.voiceBtn.innerHTML = '<i class="fas fa-microphone"></i> Voice Input';
        this.updateStatus('🎤 Voice input ready - Click to speak', 'info');
    }

    openAslModal() {
        this.aslModal.style.display = 'flex';
        this.aslInput.focus();
        this.updateStatus('🤟 ASL Learning opened - Enter a letter or word to learn!', 'info');
        
        // Auto-start with letter A if no input
        if (!this.aslInput.value.trim()) {
            this.aslInput.value = 'A';
            this.learnAslSign();
        }
    }

    closeAslModalHandler() {
        this.aslModal.style.display = 'none';
        
        // Reset the modal content
        this.aslResult.innerHTML = `
            <div class="asl-placeholder">
                <i class="fas fa-hands"></i>
                <p>Enter a letter or word to learn its American Sign Language!</p>
            </div>
        `;
        
        // Stop auto-learning if it's running
        if (this.isAutoAsl) {
            this.stopAutoAsl();
        }
        
        // Reset to first letter
        this.currentAslLetter = 0;
        this.aslInput.value = 'A';
        
        this.updateStatus('ASL Learning closed', 'info');
    }

    async learnAslSign() {
        const input = this.aslInput.value.trim();
        if (!input) {
            this.updateStatus('❌ Please enter a letter or word to learn', 'error');
            return;
        }

        // Show loading state
        this.aslResult.innerHTML = `
            <div class="asl-loading">
                <i class="fas fa-spinner"></i>
                <p>🤟 Learning ASL sign for "${input}"...</p>
            </div>
        `;

        this.learnAslBtn.disabled = true;
        this.learnAslBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Learning...';

        try {
            const aslLesson = await this.getAslLesson(input);
            
            // Safely extract the lesson text
            if (aslLesson && aslLesson.candidates && aslLesson.candidates[0] && 
                aslLesson.candidates[0].content && aslLesson.candidates[0].content.parts && 
                aslLesson.candidates[0].content.parts[0] && aslLesson.candidates[0].content.parts[0].text) {
                const lessonText = aslLesson.candidates[0].content.parts[0].text;
                this.displayAslLesson(lessonText);
                this.updateStatus(`🤟 ASL lesson loaded for "${input}"`, 'success');
            } else {
                throw new Error('Invalid lesson format received from API');
            }
        } catch (error) {
            console.error('ASL Learning error:', error);
            this.displayAslError(error.message);
            this.updateStatus('❌ Error learning ASL sign', 'error');
        } finally {
            this.learnAslBtn.disabled = false;
            this.learnAslBtn.innerHTML = '<i class="fas fa-graduation-cap"></i> Learn Sign';
        }
    }

    async getAslLesson(input) {
        // First try the new Gemini API endpoint
        const prompt = `Teach American Sign Language (ASL) for the letter/word "${input}". 
        Provide a detailed, step-by-step guide that includes:
        1. Hand position and finger placement
        2. Movement description
        3. Common mistakes to avoid
        4. Practice tips
        5. Visual description of the sign
        
        Make it educational and accessible for deaf and hard-of-hearing learners. 
        Include specific details about finger positioning, palm orientation, and any movements.
        Format the response in a clear, structured way with numbered steps.`;

        try {
            // Try the newer Gemini API endpoint first
            const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${this.geminiApiKey}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{
                            text: prompt
                        }]
                    }]
                })
            });

            if (!response.ok) {
                // If the new endpoint fails, try the old one
                const fallbackResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${this.geminiApiKey}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        contents: [{
                            parts: [{
                                text: prompt
                            }]
                        }]
                    })
                });

                if (!fallbackResponse.ok) {
                    throw new Error(`API request failed: ${fallbackResponse.status}`);
                }
                return await fallbackResponse.json();
            }

            const data = await response.json();
            
            if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
                throw new Error('Invalid response from Gemini API');
            }
            
            return data;
        } catch (error) {
            // If API fails, provide a fallback ASL lesson
            console.warn('Gemini API failed, using fallback ASL lesson:', error);
            return this.getFallbackAslLesson(input);
        }
    }

    getFallbackAslLesson(input) {
        const letter = input.toUpperCase();
        const aslLessons = {
            'A': {
                description: 'Make a fist with your thumb pointing up and to the side',
                steps: [
                    'Start with your hand in a relaxed position',
                    'Make a fist with all fingers curled',
                    'Extend your thumb straight up and slightly to the side',
                    'Keep your palm facing forward',
                    'Hold the position steady'
                ],
                tips: ['Keep your thumb straight and strong', 'Don\'t let other fingers stick out']
            },
            'B': {
                description: 'Hold all fingers straight and close together',
                steps: [
                    'Start with your hand open',
                    'Keep all four fingers (index, middle, ring, pinky) straight and together',
                    'Tuck your thumb against your palm',
                    'Keep fingers parallel to each other',
                    'Hold the position steady'
                ],
                tips: ['Keep fingers straight and close together', 'Thumb should be tucked, not extended']
            },
            'C': {
                description: 'Form a C-shape with your hand',
                steps: [
                    'Start with your hand open',
                    'Curve your fingers to form a C-shape',
                    'Keep your thumb and fingers separated',
                    'Make sure the opening is clear',
                    'Hold the position steady'
                ],
                tips: ['Make sure the C-shape is clear and recognizable', 'Don\'t close the fingers too much']
            }
        };

        const lesson = aslLessons[letter] || {
            description: `Sign for the letter ${letter}`,
            steps: [
                'Start with your hand in a comfortable position',
                'Form the specific sign for this letter',
                'Hold the position clearly',
                'Practice the movement if needed',
                'Repeat to reinforce learning'
            ],
            tips: ['Practice slowly at first', 'Focus on clarity over speed', 'Use a mirror to check your form']
        };

        return {
            candidates: [{
                content: {
                    parts: [{
                        text: `🤟 ASL Lesson for Letter "${letter}"

📝 Description: ${lesson.description}

📋 Step-by-Step Instructions:
${lesson.steps.map((step, i) => `${i + 1}. ${step}`).join('\n')}

💡 Practice Tips:
${lesson.tips.map(tip => `• ${tip}`).join('\n')}

🎯 Common Mistakes to Avoid:
• Don't rush the sign - clarity is more important than speed
• Make sure your hand position is correct
• Practice in front of a mirror to check your form
• Keep your movements smooth and controlled

🔄 Practice Exercise:
1. Start with your hand in the neutral position
2. Slowly form the sign for "${letter}"
3. Hold it for 3 seconds
4. Return to neutral position
5. Repeat 5 times

Remember: American Sign Language is a beautiful and expressive language. Take your time to learn each sign properly!`
                    }]
                }
            }]
        };
    }

    getLetterImage(letter) {
        // Create image element for the letter
        const imagePath = `data/${letter.toUpperCase()}.jpg`;
        return `
            <div class="asl-letter-image-container">
                <img src="${imagePath}" 
                     alt="Letter ${letter}" 
                     class="asl-letter-image"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div class="asl-letter-placeholder" style="display: none;">
                    <i class="fas fa-image"></i>
                    <span>${letter}</span>
                </div>
            </div>
        `;
    }

    displayAslLesson(lessonText) {
        // Ensure lessonText is a string
        if (typeof lessonText !== 'string') {
            console.error('displayAslLesson received non-string:', lessonText);
            this.displayAslError('Invalid lesson format received');
            return;
        }
        
        // Get the current letter being taught
        const currentLetter = this.aslAlphabet[this.currentAslLetter];
        const letterImage = this.getLetterImage(currentLetter);
        
        // Parse the lesson text and format it nicely
        const lines = lessonText.split('\n').filter(line => line.trim());
        let formattedLesson = '';

        lines.forEach((line, index) => {
            const trimmedLine = line.trim();
            
            if (trimmedLine.match(/^\d+\./)) {
                // Numbered step
                const stepNumber = trimmedLine.match(/^(\d+)\./)[1];
                const stepContent = trimmedLine.replace(/^\d+\.\s*/, '');
                formattedLesson += `
                    <div class="asl-step">
                        <div class="asl-step-number">${stepNumber}</div>
                        <div class="asl-step-content">
                            <h5>${stepContent}</h5>
                        </div>
                    </div>
                `;
            } else if (trimmedLine.startsWith('**') && trimmedLine.endsWith('**')) {
                // Bold heading
                const heading = trimmedLine.replace(/\*\*/g, '');
                formattedLesson += `<h4>${heading}</h4>`;
            } else if (trimmedLine.length > 0) {
                // Regular paragraph
                formattedLesson += `<p>${trimmedLine}</p>`;
            }
        });

        this.aslResult.innerHTML = `
            <div class="asl-lesson">
                <div class="asl-lesson-header">
                    <h4>🤟 ASL Sign for Letter "${currentLetter}"</h4>
                    <div class="lesson-progress">
                        <span class="current-letter">${currentLetter}</span>
                        <span class="progress-text">${this.currentAslLetter + 1} of ${this.aslAlphabet.length}</span>
                    </div>
                </div>
                
                <div class="asl-visual-section">
                    <h5>📸 Visual Reference - Letter "${currentLetter}"</h5>
                    ${letterImage}
                    <p class="image-caption">This is how the letter "${currentLetter}" looks in written form</p>
                </div>
                
                <div class="asl-instructions">
                    <h5>✋ Sign Language Instructions</h5>
                    <div class="asl-steps">
                        ${formattedLesson}
                    </div>
                </div>
                
                <div class="asl-tips">
                    <h5>💡 Practice Tips:</h5>
                    <ul>
                        <li>Look at the letter image above to understand the shape</li>
                        <li>Practice slowly and focus on accuracy</li>
                        <li>Use a mirror to check your hand position</li>
                        <li>Practice with a partner if possible</li>
                        <li>Repeat the sign multiple times to build muscle memory</li>
                    </ul>
                </div>
            </div>
        `;
    }

    displayAslError(message) {
        this.aslResult.innerHTML = `
            <div class="asl-error">
                <i class="fas fa-exclamation-triangle"></i>
                <h4>Error Learning ASL Sign</h4>
                <p>${message}</p>
                <p>Please try again or check your internet connection.</p>
            </div>
        `;
    }

    updateStatus(message, type = 'info') {
        const icons = {
            info: 'fas fa-info-circle',
            success: 'fas fa-check-circle',
            warning: 'fas fa-exclamation-triangle',
            error: 'fas fa-times-circle'
        };

        const colors = {
            info: '#6c757d',
            success: '#27ae60',
            warning: '#f39c12',
            error: '#e74c3c'
        };

        this.statusText.innerHTML = `
            <i class="${icons[type]}" style="color: ${colors[type]}"></i>
            ${message}
        `;
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    getTimeAgo(date) {
        const now = new Date();
        const diff = now - date;
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days} day${days !== 1 ? 's' : ''} ago`;
        if (hours > 0) return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
        if (minutes > 0) return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
        return 'Just now';
    }

    exportWords() {
        if (this.words.length === 0) {
            this.updateStatus('⚠️ No words to export', 'warning');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            totalWords: this.words.length,
            totalLetters: this.words.reduce((sum, word) => sum + word.letters.length, 0),
            words: this.words.map(word => ({
                text: word.text,
                letters: word.letters,
                timestamp: word.timestamp.toISOString(),
                imageCount: word.images.filter(img => img.exists).length,
                missingImages: word.images.filter(img => !img.exists).map(img => img.letter)
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `alphabet-words-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.updateStatus('📁 Words exported successfully', 'success');
    }

    // ASL Navigation Methods
    previousAslLetter() {
        this.currentAslLetter = Math.max(0, this.currentAslLetter - 1);
        const letter = this.aslAlphabet[this.currentAslLetter];
        this.aslInput.value = letter;
        this.learnAslSign();
        this.updateStatus(`🤟 Learning ASL for letter: ${letter}`, 'info');
    }

    nextAslLetter() {
        this.currentAslLetter = Math.min(this.aslAlphabet.length - 1, this.currentAslLetter + 1);
        const letter = this.aslAlphabet[this.currentAslLetter];
        this.aslInput.value = letter;
        this.learnAslSign();
        this.updateStatus(`🤟 Learning ASL for letter: ${letter}`, 'info');
    }

    toggleAutoAsl() {
        if (this.isAutoAsl) {
            this.stopAutoAsl();
        } else {
            this.startAutoAsl();
        }
    }

    startAutoAsl() {
        this.isAutoAsl = true;
        this.autoAslBtn.innerHTML = '<i class="fas fa-pause"></i> Stop Auto';
        this.autoAslBtn.classList.remove('btn-success');
        this.autoAslBtn.classList.add('btn-warning');
        this.updateStatus('🤟 Auto ASL learning started - Going through alphabet!', 'success');
        
        // Start with current letter
        this.learnAslSign();
        
        // Auto-advance every 10 seconds
        this.autoAslInterval = setInterval(() => {
            this.nextAslLetter();
        }, 10000);
    }

    stopAutoAsl() {
        this.isAutoAsl = false;
        this.autoAslBtn.innerHTML = '<i class="fas fa-play"></i> Auto Learn';
        this.autoAslBtn.classList.remove('btn-warning');
        this.autoAslBtn.classList.add('btn-success');
        this.updateStatus('🤟 Auto ASL learning stopped', 'info');
        
        if (this.autoAslInterval) {
            clearInterval(this.autoAslInterval);
            this.autoAslInterval = null;
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AlphabetDisplayApp();
});

// Add some utility functions for better UX
document.addEventListener('keydown', (e) => {
    // Global keyboard shortcuts
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        document.getElementById('textInput').focus();
    }
});

// Add smooth scrolling for better UX
document.documentElement.style.scrollBehavior = 'smooth';
