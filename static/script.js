async function askQuestion() {
    const question = document.getElementById("question").value.trim();
    if (!question) return;

    // Add user's message to chat
    addMessage("user", question);
    document.getElementById("question").value = "";

    // Call API
    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: question })
    });

    const data = await response.json();
    addMessage("ai", data.answer || "Sorry, I couldn't find an answer.");
}

function addMessage(sender, text) {
    const chatBox = document.getElementById("chat-box");
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender === "user" ? "user-message" : "ai-message");
    messageDiv.innerText = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
