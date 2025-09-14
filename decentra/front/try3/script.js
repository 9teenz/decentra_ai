document.addEventListener('DOMContentLoaded', function() {
    const notificationButton = document.getElementById('showNotification');
    const notificationContainer = document.getElementById('notificationContainer');
    let notificationCount = 0;
    
    // Массив с уведомлениями
    const notifications = [
        {
            title: 'Новое предложение',
            message: 'Карта с увеличенным кешбэком до 10% на кафе и рестораны',
            icon: 'fa-gift',
            delay: 5000
        },
    ];
    
    // Функция создания уведомления
    function createNotification(title, message, icon) {
        notificationCount++;
        const notificationId = 'notification-' + notificationCount;
        
        const notification = document.createElement('div');
        notification.className = 'push-notification';
        notification.id = notificationId;
        
        const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="fas ${icon}"></i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
                <div class="notification-time">${currentTime}</div>
            </div>
            <button class="close-notification" onclick="closeNotification('${notificationId}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        notificationContainer.appendChild(notification);
        
        // Показываем уведомление с анимацией
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        // Автоматическое закрытие через 7 секунд
        setTimeout(() => {
            if (document.getElementById(notificationId)) {
                closeNotification(notificationId);
            }
        }, 7000);
    }
    
    // Функция закрытия уведомления
    window.closeNotification = function(id) {
        const notification = document.getElementById(id);
        if (notification) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }
    };
    

    // Обработчик кнопки "Показать уведомление"
    notificationButton.addEventListener('click', function() {
        fetch('clients_combined2.json') // путь к файлу с текстом
            .then(response => response.text())
            .then(text => {
                createNotification(
                    'Новое уведомление',
                    text, // сюда подставляется содержимое файла
                    'fa-gift'
                );
            })
        .catch(error => {
            console.error('Ошибка загрузки файла:', error);
            createNotification(
                'Ошибка',
                'Не удалось загрузить сообщение',
                'fa-exclamation-triangle'
            );
        });
});
    // notificationButton.addEventListener('click', function() {
    //     const randomNotification = notifications[Math.floor(Math.random() * notifications.length)];
    //     createNotification(
    //         randomNotification.title,
    //         randomNotification.message,
    //         randomNotification.icon
    //     );
    // });
    
    // Автоматическая отправка уведомлений с интервалом
    
    // Запрос разрешения на уведомления (при первом посещении)
    if ('Notification' in window && Notification.permission === 'default') {
        setTimeout(() => {
            Notification.requestPermission();
        }, 2000);
    }
});

let clientsData = [];

async function loadClients() {
  try {
    const response = await fetch("clients_combined2.json");
    clientsData = await response.json();
    showClient(1); // по умолчанию клиент с ID=1
  } catch (error) {
    console.error("Ошибка загрузки данных:", error);
  }
}

function showClient(id) {
  const client = clientsData.find(c => c.id === id);
  if (!client) {
    alert(`Клиент с ID ${id} не найден`);
    return;
  }

  document.querySelector(".client-name").textContent = client.name;
  document.querySelector(".client-status").textContent = client.status;
  document.getElementById("ageid").textContent = client.age;
  document.getElementById("city").textContent = client.city;
  document.querySelector(".balance").textContent = client.balance;
  document.getElementById("productid").textContent = client.product;
  document.getElementById("top1").textContent = client.top_cats[0];
  document.getElementById("top2").textContent = client.top_cats[1];
  document.getElementById("top3").textContent = client.top_cats[2];
  document.getElementById("top4").textContent = client.top_cats[3];
  document.getElementById("top5").textContent = client.top_cats[4];
  document.getElementById("offertext").textContent = client.push_notification;
}

function searchClient() {
  const id = Number(document.getElementById("searchInput").value);
  if (id) {
    showClient(id);
  }
}

window.onload = loadClients;
