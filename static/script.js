// scripts.js (Güncellenmiş Hali)

const form = document.getElementById('chat-form');
const messages = document.getElementById('messages');
const status = document.getElementById('status');
const textarea = document.getElementById('question');

// GÜNCELLEME: Konuşma geçmişini saklamak için bir değişken oluşturuldu.
let conversationHistory = [];

function addMessage(text, who = 'bot') {
    const div = document.createElement('div');
    div.className = 'msg ' + who;
    div.innerHTML = marked.parse(text); // marked.js kütüphanesinin markdown'ı HTML'e çevirdiğini varsayıyoruz.
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight; // Her yeni mesajda en alta kaydır
}

function addThinkingIndicator() {
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'msg bot'; // 'bot' stiliyle uyumlu olsun
    thinkingDiv.id = 'thinking-indicator';
    thinkingDiv.innerHTML = `
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `; // Daha basit bir "yazıyor" animasyonu
    messages.appendChild(thinkingDiv);
    messages.scrollTop = messages.scrollHeight;
}

function removeThinkingIndicator() {
    const thinkingIndicator = document.getElementById('thinking-indicator');
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}

async function sendMessage() {
    const q = textarea.value.trim();
    if (!q) return;

    addMessage(q, 'user');
    textarea.value = '';
    status.textContent = 'Thinking...';

    addThinkingIndicator();

    try {
        // GÜNCELLEME: fetch isteğinin body'sine 'history' eklendi.
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: q,
                history: conversationHistory
            })
        });

        if (!res.ok) {
            const errorData = await res.text();
            throw new Error(`Server error: ${res.status} - ${errorData}`);
        }

        const data = await res.json();
        const botAnswer = data.answer || 'No response';

        removeThinkingIndicator();
        addMessage(botAnswer, 'bot');

        // GÜNCELLEME: Başarılı bir cevaptan sonra konuşma geçmişini güncelle.
        conversationHistory.push({
            question: q,
            answer: botAnswer
        });

    } catch (err) {
        removeThinkingIndicator();
        addMessage(`[Error] ${err.message}`, 'bot');
    } finally {
        status.textContent = 'Ready';
        textarea.focus();
    }
}

// Form submit event
form.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
});

// Enter tuşu ile gönderme (Shift+Enter ile yeni satır)
textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Tema geçişi sistemi ve diğer UI kodları aynı kalabilir...
const themeToggle = document.getElementById('theme-toggle');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const currentTheme = localStorage.getItem('theme') || (prefersDark ? 'dark' : 'light');

document.body.classList.toggle('dark', currentTheme === 'dark');
themeToggle.textContent = currentTheme === 'dark' ? '🌞' : '🌙';

themeToggle.addEventListener('click', () => {
    const isDark = document.body.classList.toggle('dark');
    themeToggle.textContent = isDark ? '🌞' : '🌙';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
});

textarea.focus();