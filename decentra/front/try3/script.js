let clientsData = [];
let currentClientId = null;
let notificationButton; // сделаем глобальной
let notificationContainer;
let notificationCount = 0;

document.addEventListener('DOMContentLoaded', function() {
    notificationButton = document.getElementById('showNotification');
    notificationContainer = document.getElementById('notificationContainer');
    
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
        
        setTimeout(() => notification.classList.add('show'), 100);
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

    // обработчик кнопки
    notificationButton.addEventListener('click', function() {
        if (!currentClientId) {
            alert("Сначала выберите клиента");
            return;
        }

        const client = clientsData.find(c => c.id === currentClientId);
        if (!client) {
            alert("Ошибка: клиент не найден");
            return;
        }

        createNotification(
            "Персональное предложение",
            client.push_notification,
            "fa-bell"
        );
    });

    // Запрос разрешения на системные уведомления
    if ('Notification' in window && Notification.permission === 'default') {
        setTimeout(() => Notification.requestPermission(), 2000);
    }
});

// загрузка клиентов
async function loadClients() {
    try {
        const response = await fetch("clients_combined2.json");
        clientsData = await response.json();
        showClient(1); // по умолчанию
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

    currentClientId = id; // сохраняем выбранного клиента

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
