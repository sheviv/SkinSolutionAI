import streamlit as st
from datetime import datetime


def init_chat():
    """Initialize chat system in session state if not already set"""
    if 'chat_active' not in st.session_state:
        st.session_state.chat_active = False

    if 'current_doctor' not in st.session_state:
        st.session_state.current_doctor = None

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = {}

    if 'chat_input' not in st.session_state:
        st.session_state.chat_input = ""

    if 'show_chat_modal' not in st.session_state:
        st.session_state.show_chat_modal = False


def start_chat_with_doctor(doctor):
    """Start a chat session with a specific doctor"""
    st.session_state.current_doctor = doctor

    # Initialize chat history for this doctor if doesn't exist
    doctor_id = doctor['name']
    if doctor_id not in st.session_state.chat_messages:
        st.session_state.chat_messages[doctor_id] = [
            {
                "sender": "doctor",
                "message": f"Здравствуйте! Я доктор {doctor['name']}, специалист в {doctor['speciality']}. Чем я могу вам помочь?",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]

    # Show chat modal
    st.session_state.show_chat_modal = True


def close_chat():
    """Close the active chat modal"""
    st.session_state.show_chat_modal = False


def add_message(message):
    """Add a message to the current chat"""
    if not st.session_state.current_doctor:
        return

    doctor_id = st.session_state.current_doctor['name']

    if doctor_id in st.session_state.chat_messages:
        st.session_state.chat_messages[doctor_id].append({
            "sender": "user",
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        # Add simple automated response from doctor (in a real app, this would be more sophisticated)
        st.session_state.chat_messages[doctor_id].append({
            "sender": "doctor",
            "message": "Спасибо за ваше сообщение. Я скоро свяжусь с вами.",
            "timestamp": datetime.now().strftime("%H:%M")
        })


def render_chat_interface():
    """Render the chat interface as a floating modal"""
    if not st.session_state.show_chat_modal or not st.session_state.current_doctor:
        return

    doctor = st.session_state.current_doctor
    doctor_id = doctor['name']

    # CSS for the chat modal
    st.markdown("""
    <style>
    .chat-modal {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        max-height: 500px;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        background: #4b9bd3;
        color: white;
        padding: 10px 15px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .chat-body {
        padding: 10px 15px;
        overflow-y: auto;
        max-height: 350px;
        flex-grow: 1;
    }
    .message {
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 15px;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user-message {
        background: #e6f2ff;
        margin-left: auto;
    }
    .doctor-message {
        background: #f1f1f1;
        margin-right: auto;
    }
    .timestamp {
        font-size: 10px;
        color: #888;
        margin-top: 4px;
    }
    .chat-footer {
        padding: 10px;
        border-top: 1px solid #eee;
    }
    .close-button {
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create modal HTML structure
    messages_html = ""
    if doctor_id in st.session_state.chat_messages:
        for msg in st.session_state.chat_messages[doctor_id]:
            msg_class = "user-message" if msg["sender"] == "user" else "doctor-message"
            messages_html += f"""
            <div class="message {msg_class}">
                {msg["message"]}
                <div class="timestamp">{msg["timestamp"]}</div>
            </div>
            """

    modal_html = f"""
    <div class="chat-modal">
        <div class="chat-header">
            <div>Чат с Доктором {doctor['name']}</div>
            <button class="close-button" onclick="document.querySelector('.chat-modal').style.display='none';">&times;</button>
        </div>
        <div class="chat-body">
            {messages_html}
        </div>
        <div class="chat-footer">
            <form id="chat-form">
                <input type="text" id="chat-input" placeholder="Введите сообщение..." style="width: 75%; padding: 8px;">
                <button type="submit" style="width: 20%; padding: 8px;">Отправить</button>
            </form>
        </div>
    </div>

    <script>
    // Make the form submit via AJAX to prevent page refresh
    document.getElementById('chat-form').addEventListener('submit', function(e) {{
        e.preventDefault();
        var input = document.getElementById('chat-input');
        var message = input.value.trim();

        if (message) {{
            // Add message to UI
            var chatBody = document.querySelector('.chat-body');
            var time = new Date().toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit'}});
            var msgHtml = `
                <div class="message user-message">
                    ${message}
                    <div class="timestamp">${time}</div>
                </div>
            `;
            chatBody.innerHTML += msgHtml;
            chatBody.scrollTop = chatBody.scrollHeight;

            // Send message to server (would integrate with backend)
            fetch('http://0.0.0.0:5000/send_message', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{message: message, doctor: "{doctor_id}"}})
            }})
            .then(response => response.json())
            .then(data => {{
                // Add response from doctor
                var respHtml = `
                    <div class="message doctor-message">
                        ${data.response}
                        <div class="timestamp">${time}</div>
                    </div>
                `;
                chatBody.innerHTML += respHtml;
                chatBody.scrollTop = chatBody.scrollHeight;
            }})
            .catch(error => {{
                // Fallback for demo - add simple response
                var respHtml = `
                    <div class="message doctor-message">
                        Спасибо за ваше сообщение. Я скоро свяжусь с вами.
                        <div class="timestamp">${time}</div>
                    </div>
                `;
                setTimeout(() => {{
                    chatBody.innerHTML += respHtml;
                    chatBody.scrollTop = chatBody.scrollHeight;
                }}, 1000);
            }});

            // Clear input
            input.value = '';
        }}
    }});
    </script>
    """

    # Insert the modal HTML
    st.markdown(modal_html, unsafe_allow_html=True)