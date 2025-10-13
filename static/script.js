const form = document.getElementById('chat-form');
const messages = document.getElementById('messages');
const status = document.getElementById('status');
const textarea = document.getElementById('question');

function addMessage(text, who = 'bot', scrollToBottom = false) {
    const div = document.createElement('div');
    div.className = 'msg ' + who;
    div.innerHTML = marked.parse(text);

    messages.appendChild(div);

    // ✅ Sadece scrollToBottom true ise en alta kaydır
    if (scrollToBottom) {
        messages.scrollTop = messages.scrollHeight;
    }
}

// 🎓 Düşünme indicator'ü ekleme fonksiyonu
function addThinkingIndicator() {
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'thinking-indicator';
    thinkingDiv.id = 'thinking-indicator';
    thinkingDiv.innerHTML = `
        <div class="graduation-cap">🎓</div>
        <span>Thinking for the best response...</span>
    `;
    messages.appendChild(thinkingDiv);
    // ✅ Düşünme indicator'ü eklenince en alta kaydır (kullanıcı görebilsin)
    messages.scrollTop = messages.scrollHeight;
}

// 🎓 Düşünme indicator'ünü kaldırma fonksiyonu
function removeThinkingIndicator() {
    const thinkingIndicator = document.getElementById('thinking-indicator');
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}

// 📤 Mesaj gönderme fonksiyonu
async function sendMessage() {
    const q = textarea.value.trim();
    if (!q) return;

    // ✅ Kullanıcı mesajını ekle ve EN ALTA KAYDIR (kullanıcı görebilsin)
    addMessage(q, 'user', true);
    textarea.value = '';
    status.textContent = 'Thinking...';

    // 🎓 Düşünme indicator'ünü göster (zaten en altdayız)
    addThinkingIndicator();

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q })
        });

        if (!res.ok) throw new Error('Server error');
        const data = await res.json();

        // 🎓 Düşünme indicator'ünü kaldır ve cevabı göster
        removeThinkingIndicator();

        // ✅ Bot yanıtını ekle ama EN ALTA KAYDIRMA!
        addMessage(data.answer || 'No response', 'bot', false);

    } catch (err) {
        // 🎓 Hata durumunda da indicator'ü kaldır
        removeThinkingIndicator();
        // ✅ Hata mesajını da en alta kaydırma
        addMessage('[Error] ' + err.message, 'bot', false);
    } finally {
        status.textContent = 'Ready';
        // ✅ Burda da en alta kaydırma yok!
    }
}

// 📝 Form submit event
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    sendMessage();
});

// ⌨️ Enter tuşu ile gönderme (Shift+Enter ile yeni satır)
textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 🌗 Tema geçişi sistemi
const themeToggle = document.getElementById('theme-toggle');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const currentTheme = localStorage.getItem('theme') || (prefersDark ? 'dark' : 'light');

// Başlangıç temasını uygula
document.body.classList.toggle('dark', currentTheme === 'dark');
themeToggle.textContent = currentTheme === 'dark' ? '🌞' : '🌙';

// Tıklama ile değiştir
themeToggle.addEventListener('click', () => {
    const isDark = document.body.classList.toggle('dark');
    themeToggle.textContent = isDark ? '🌞' : '🌙';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
});

// Textarea'ya focus ver
textarea.focus();