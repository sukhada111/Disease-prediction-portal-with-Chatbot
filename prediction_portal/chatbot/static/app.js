class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;
        console.log(text1);
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        fetch('http://127.0.0.1:8000/chatres/', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            let msg2 = { name: "Healthino", message: r.answer};
            console.log(msg2);
            // console.log(r.link);
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Healthino")
            {
                // html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
                // console.log(item.message[1]);
                // let result = item.message.indexOf("https://www.youtube.com/");
                // if(result!=-1){
                //     html+='<div class="messages__item messages__item--visitor"><a href ='+ item.message.slice(result,item.message.length) +'></a></div>';
                // }
                function rep(text) {
                    // Put the URL to variable $1 and Domain name
                    // to $3 after visiting the URL
                    var Rexp =
        /(\b(https?|ftp|file):\/\/([-A-Z0-9+&@#%?=~_|!:,.;]*)([-A-Z0-9+&@#%?\/=~_|!:,.;]*)[-A-Z0-9+&@#\/%=~_|])/ig;
                     
                    // Replacing the RegExp content by HTML element
                    return text.replace(Rexp,
                            "<a href='$1' target='_blank'>$3</a>");
                }
                let str_html= rep(item.message)
                console.log(str_html)
                html += '<div class="messages__item messages__item--visitor">' + str_html + '</div>';
                
                
        
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' +  item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}


const chatbox = new Chatbox();
chatbox.display();