// --- BIẾN TOÀN CỤC & CẤU HÌNH ---
let userLocation = null;
let map = null;
let routingControl = null;
let animatedMarker = null;
let currentTravelMode = 'car';
let currentDestination = null;
let currentDetailRestaurant = null;

// (MỚI) Quản lý trạng thái đăng nhập
let currentUserId = null; 
let currentUsername = null;
let savedRestaurants = []; // Sẽ được tải từ API

const API_URL = "http://127.0.0.1:5000";

// --- ICON CHO NÚT LƯU ---
const ICON_SAVE = {
    unfilled: `<svg xmlns="http://www.w.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6 mr-2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z" /></svg>`,
    filled: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6 mr-2 text-red-500"><path d="M11.645 20.91l-.007-.003-.022-.012a15.247 15.247 0 01-.383-.218 25.18 25.18 0 01-4.244-3.17C4.688 15.36 2.25 12.174 2.25 8.25 2.25 5.322 4.714 3 7.688 3A5.5 5.5 0 0112 5.052 5.5 5.5 0 0116.313 3c2.973 0 5.437 2.322 5.437 5.25 0 3.925-2.438 7.111-4.739 9.256a25.175 25.175 0 01-4.244 3.17 15.247 15.247 0 01-.383.218l-.022.012-.007.004-.003.001a.752.752 0 01-.704 0l-.003-.001z" /></svg>`
};

// --- ICON BẢN ĐỒ (Giữ nguyên) ---
const ICONS_SVG = { /* ... (giữ nguyên các icon car, bike, walk) ... */ 
    car: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8 text-yellow-500 drop-shadow-lg"><path d="M5.507 8.493l-.434 2.598A3.75 3.75 0 008.25 15h7.5a3.75 3.75 0 003.178-3.909l-.434-2.598a.75.75 0 00-.73-.643H6.237a.75.75 0 00-.73.643zM12 3a.75.75 0 00-.75.75v.755a3 3 0 00-1.652.89l-.421-.422a.75.75 0 10-1.06 1.06l.421.422a3 3 0 00-.89 1.652H6.75a.75.75 0 00-.75.75v1.5c0 .414.336.75.75.75h.755a3 3 0 00.89 1.652l-.422.421a.75.75 0 101.06 1.06l.422-.421a3 3 0 001.652.89v.755a.75.75 0 001.5 0v-.755a3 3 0 001.652-.89l.421.422a.75.75 0 101.06-1.06l-.421-.422a3 3 0 00.89-1.652h.755a.75.75 0 00.75-.75v-1.5a.75.75 0 00-.75-.75h-.755a3 3 0 00-.89-1.652l.422-.421a.75.75 0 10-1.06-1.06l-.422.421a3 3 0 00-1.652-.89V3.75A.75.75 0 0012 3zM12 7.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3z" /></svg>`,
    bike: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8 text-gray-800"><path fill-rule="evenodd" d="M9.164 1.832a.75.75 0 01.515.243l3.375 4.125a.75.75 0 01-.243 1.031l-.478.359a.75.75 0 01-.986-.145l-2.03-3.248a.75.75 0 00-1.295.808l2.585 4.137a.75.75 0 01-.33 1.02l-.478.358a.75.75 0 01-.986-.145L7.5 7.152v2.1a.75.75 0 01-1.5 0v-3.41a.75.75 0 01.243-1.031l3.375-4.125a.75.75 0 01.546-.243zM14.5 2.25a2 2 0 100 4 2 2 0 000-4z" clip-rule="evenodd" /><path d="M11.25 11.25a.75.75 0 01.75-.75h1.5a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0v-1.5h-.75a.75.75 0 01-.75-.75z" /><path d="M3.75 13.5a.75.75 0 000 1.5h10.536l-1.34 2.233a.75.75 0 101.248.746l2.122-3.536a.75.75 0 000-.746l-2.122-3.536a.75.75 0 10-1.248.746L14.286 15H3.75z" /><path d="M15.5 12.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM4 12.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5z" /></svg>`,
    walk: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8 text-gray-800"><path fill-rule="evenodd" d="M11.47 2.47a.75.75 0 011.06 0l4.5 4.5a.75.75 0 01-1.06 1.06l-3.22-3.22V16.5a.75.75 0 01-1.5 0V4.81L8.03 8.03a.75.75 0 01-1.06-1.06l4.5-4.5zM12 18a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5z" clip-rule="evenodd" /><path d="M6.75 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5zM3 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5zM18.75 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5zM15 19.5a.75.75 0 00-1.5 0v2.25a.75.75 0 001.5 0V19.5z" /></svg>`
};
const CUSTOM_SPEEDS_KMH = { car: 30, bike: 15, walk: 4 };
const TRAVEL_MODES = [
    { id: 'car', label: 'Xe hơi', icon: ICONS_SVG.car, profile: 'driving' },
    { id: 'bike', label: 'Xe đạp', icon: ICONS_SVG.bike, profile: 'bicycle' },
    { id: 'walk', label: 'Đi bộ', icon: ICONS_SVG.walk, profile: 'foot' }
];

// --- LẤY CÁC THÀNH PHẦN DOM (CHUNG) ---
let getLocationBtn, locationStatus, restaurantListEl, searchInput, searchButton, searchStatus;

// --- LẤY DOM CHO MODAL BẢN ĐỒ ---
let mapModal, closeMapModalBtn, mapModalTitle, startAnimationBtn, summaryDistance, summaryTime, mapLoader, travelModeSelector;

// --- LẤY DOM CHO MODAL CHI TIẾT ---
let detailModal, closeDetailModalBtn, detailModalTitle, detailModalImage, detailModalAddress, detailModalSaveBtn, detailModalSaveIcon, detailModalSaveText, detailModalShareBtn, detailModalNavigateBtn;

// --- (MỚI) LẤY DOM CHO AUTH VÀ MODAL LOGIN ---
let authContainer, loginButton, userGreeting, usernameDisplay, logoutButton;
let loginModal, closeLoginModalBtn, loginForm, registerForm, loginStatus, registerStatus;

// --- KHỞI TẠO KHI TẢI TRANG ---
document.addEventListener('DOMContentLoaded', () => {
    // Gán tất cả các biến DOM
    assignDomElements();
    
    // Kiểm tra xem user đã đăng nhập từ trước chưa (dùng localStorage)
    checkLoginStatus();
    
    // Gắn các sự kiện
    addCoreEventListeners();
});

// --- (MỚI) Hàm gán DOM ---
function assignDomElements() {
    getLocationBtn = document.getElementById('getLocationBtn');
    locationStatus = document.getElementById('locationStatus');
    restaurantListEl = document.getElementById('restaurant-list');
    searchInput = document.getElementById('search-input');
    searchButton = document.getElementById('search-button');
    searchStatus = document.getElementById('search-status');
    mapModal = document.getElementById('mapModal');
    closeMapModalBtn = document.getElementById('closeMapModal');
    mapModalTitle = document.getElementById('mapModalTitle');
    startAnimationBtn = document.getElementById('startAnimationBtn');
    summaryDistance = document.getElementById('summary-distance');
    summaryTime = document.getElementById('summary-time');
    mapLoader = document.getElementById('map-loader');
    travelModeSelector = document.getElementById('travel-mode-selector');
    detailModal = document.getElementById('detailModal');
    closeDetailModalBtn = document.getElementById('closeDetailModal');
    detailModalTitle = document.getElementById('detailModalTitle');
    detailModalImage = document.getElementById('detailModalImage');
    detailModalAddress = document.getElementById('detailModalAddress');
    detailModalSaveBtn = document.getElementById('detailModalSaveBtn');
    detailModalSaveIcon = document.getElementById('detailModalSaveIcon');
    detailModalSaveText = document.getElementById('detailModalSaveText');
    detailModalShareBtn = document.getElementById('detailModalShareBtn');
    detailModalNavigateBtn = document.getElementById('detailModalNavigateBtn');
    authContainer = document.getElementById('auth-container');
    loginButton = document.getElementById('loginButton');
    userGreeting = document.getElementById('user-greeting');
    usernameDisplay = document.getElementById('username-display');
    logoutButton = document.getElementById('logoutButton');
    loginModal = document.getElementById('loginModal');
    closeLoginModalBtn = document.getElementById('closeLoginModal');
    loginForm = document.getElementById('loginForm');
    registerForm = document.getElementById('registerForm');
    loginStatus = document.getElementById('loginStatus');
    registerStatus = document.getElementById('registerStatus');
}

// --- HÀM GẮN SỰ KIỆN CỐ ĐỊNH ---
function addCoreEventListeners() {
    getLocationBtn.addEventListener('click', handleGetLocation);
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keyup', (e) => (e.key === 'Enter') && performSearch());
    closeMapModalBtn.addEventListener('click', closeMapModal);
    mapModal.addEventListener('click', (e) => (e.target === mapModal) && closeMapModal());
    closeDetailModalBtn.addEventListener('click', closeDetailModal);
    detailModal.addEventListener('click', (e) => (e.target === detailModal) && closeDetailModal());
    detailModalNavigateBtn.addEventListener('click', handleNavigateFromDetail);
    detailModalSaveBtn.addEventListener('click', handleSaveClick);
    detailModalShareBtn.addEventListener('click', handleShareClick);

    // (MỚI) Sự kiện cho Login/Register
    loginButton.addEventListener('click', openLoginModal);
    logoutButton.addEventListener('click', handleLogout);
    closeLoginModalBtn.addEventListener('click', closeLoginModal);
    loginModal.addEventListener('click', (e) => (e.target === loginModal) && closeLoginModal());
    loginForm.addEventListener('submit', handleLoginSubmit);
    registerForm.addEventListener('submit', handleRegisterSubmit);
}

// --- (MỚI) CÁC HÀM XỬ LÝ AUTH ---
function checkLoginStatus() {
    const userId = localStorage.getItem('currentUserId');
    const username = localStorage.getItem('currentUsername');
    
    if (userId && username) {
        // Nếu có thông tin trong localStorage, coi như đã đăng nhập
        loginSuccess(userId, username);
    }
}

async function loginSuccess(userId, username) {
    currentUserId = userId;
    currentUsername = username;

    // Lưu vào localStorage để "ghi nhớ"
    localStorage.setItem('currentUserId', userId);
    localStorage.setItem('currentUsername', username);

    // Cập nhật UI
    usernameDisplay.textContent = username;
    authContainer.classList.add('hidden');
    userGreeting.classList.remove('hidden');

    // Tải danh sách đã lưu của user
    await loadSavedRestaurantsFromServer();
    
    // Đóng modal
    closeLoginModal();
}

function handleLogout() {
    // Xóa trạng thái
    currentUserId = null;
    currentUsername = null;
    savedRestaurants = [];
    
    // Xóa localStorage
    localStorage.removeItem('currentUserId');
    localStorage.removeItem('currentUsername');

    // Cập nhật UI
    authContainer.classList.remove('hidden');
    userGreeting.classList.add('hidden');
}

function openLoginModal() {
    loginStatus.textContent = '';
    registerStatus.textContent = '';
    loginModal.classList.remove('hidden');
    setTimeout(() => loginModal.classList.remove('opacity-0'), 10);
}

function closeLoginModal() {
    loginModal.classList.add('opacity-0');
    setTimeout(() => loginModal.classList.add('hidden'), 300);
}

// (MỚI) GỌI API ĐĂNG NHẬP
async function handleLoginSubmit(e) {
    e.preventDefault();
    loginStatus.textContent = 'Đang đăng nhập...';
    
    const username = loginForm.username.value;
    const password = loginForm.password.value;

    try {
        const response = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            loginSuccess(data.user_id, data.username);
        } else {
            loginStatus.textContent = data.message;
        }
    } catch (err) {
        loginStatus.textContent = 'Lỗi kết nối máy chủ.';
    }
}

// (MỚI) GỌI API ĐĂNG KÝ
async function handleRegisterSubmit(e) {
    e.preventDefault();
    registerStatus.textContent = 'Đang đăng ký...';
    
    const username = registerForm.username.value;
    const password = registerForm.password.value;

    try {
        const response = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            registerStatus.style.color = 'green';
            registerStatus.textContent = 'Đăng ký thành công! Vui lòng đăng nhập.';
            registerForm.reset();
        } else {
            registerStatus.style.color = 'red';
            registerStatus.textContent = data.message;
        }
    } catch (err) {
        registerStatus.style.color = 'red';
        registerStatus.textContent = 'Lỗi kết nối máy chủ.';
    }
}


// --- (MỚI) HÀM TẢI DANH SÁCH ĐÃ LƯU TỪ SERVER ---
async function loadSavedRestaurantsFromServer() {
    if (!currentUserId) return; // Chỉ tải khi đã đăng nhập
    
    try {
        const response = await fetch(`${API_URL}/get_saved?user_id=${currentUserId}`);
        const data = await response.json();
        if (data.success) {
            savedRestaurants = data.saved_items;
        } else {
            console.error("Lỗi khi tải danh sách đã lưu:", data.message);
        }
    } catch (err) {
        console.error("Lỗi kết nối khi tải danh sách đã lưu:", err);
    }
}

// --- HÀM XỬ LÝ VỊ TRÍ ---
function handleGetLocation() {
    // ... (Giữ nguyên code handleGetLocation)
    if ("geolocation" in navigator) {
        locationStatus.textContent = "Đang xác định vị trí...";
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                locationStatus.textContent = `Vị trí của bạn: ${userLocation.lat.toFixed(5)}, ${userLocation.lng.toFixed(5)}`;
                locationStatus.style.color = 'green';
            },
            (err) => {
                userLocation = null;
                locationStatus.textContent = "Lỗi: Không thể lấy vị trí của bạn.";
                locationStatus.style.color = 'red';
            }
        );
    }
}

// --- HÀM TÌM KIẾM ---
async function performSearch() {
    // ... (Giữ nguyên code performSearch)
    const query = searchInput.value.trim();
    if (query === "") {
        searchStatus.textContent = "Vui lòng nhập từ khóa tìm kiếm.";
        searchStatus.style.color = 'red';
        return;
    }
    searchStatus.textContent = "Đang tìm kiếm...";
    searchStatus.style.color = 'gray';
    restaurantListEl.innerHTML = ''; 

    try {
        const response = await fetch(`${API_URL}/search?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error(`Lỗi máy chủ: ${response.statusText}`);
        const results = await response.json();
        
        if (results.length === 0) {
            searchStatus.textContent = "Không tìm thấy kết quả nào phù hợp.";
            searchStatus.style.color = 'gray';
        } else {
            searchStatus.textContent = `Tìm thấy ${results.length} kết quả.`;
            searchStatus.style.color = 'green';
            renderRestaurantList(results);
        }
    } catch (error) {
        console.error("Lỗi khi gọi API:", error);
        searchStatus.textContent = "Lỗi: Không thể kết nối đến máy chủ tìm kiếm.";
        searchStatus.style.color = 'red';
    }
}

// --- (ĐÃ CẬP NHẬT) HÀM RENDER DANH SÁCH ---
function renderRestaurantList(restaurants) {
    // ... (Giữ nguyên code renderRestaurantList)
    restaurantListEl.innerHTML = ''; 
    restaurants.forEach((r, index) => {
        const imageUrl = r.image_src 
                       ? r.image_src 
                       : 'https://placehold.co/600x400/e2e8f0/64748b?text=Không+có+ảnh';
        
        const card = document.createElement('div');
        card.className = "bg-white rounded-lg shadow-md overflow-hidden flex flex-col restaurant-item-appear cursor-pointer hover:shadow-lg transition-shadow duration-200";
        card.style.animationDelay = `${index * 50}ms`; 

        card.dataset.id = r.name; 
        card.dataset.restaurantData = JSON.stringify(r); 

        card.innerHTML = `
            <img src="${imageUrl}" alt="Ảnh quán ${r.name}" class="w-full h-48 object-cover" onerror="this.src='https://placehold.co/600x400/e2e8f0/64748b?text=Lỗi+tải+ảnh';">
            <div class="p-4 flex flex-col flex-grow">
                <h2 class="text-lg font-semibold text-gray-800 flex-grow">${r.name}</h2>
                <p class="text-sm text-gray-600 mt-1 mb-4">${r.address || 'Không có địa chỉ'}</p>
            </div>
        `;
        
        card.addEventListener('click', handleCardClick);
        restaurantListEl.appendChild(card);
    });
}

// --- HÀM XỬ LÝ KHI NHẤP VÀO THẺ (CARD) ---
function handleCardClick(event) {
    // ... (Giữ nguyên code handleCardClick)
    const card = event.currentTarget; 
    const restaurantData = JSON.parse(card.dataset.restaurantData);
    openDetailModal(restaurantData);
}

// --- CÁC HÀM CỦA MODAL CHI TIẾT ---
function openDetailModal(restaurant) {
    currentDetailRestaurant = restaurant; 

    detailModalTitle.textContent = restaurant.name;
    detailModalAddress.textContent = restaurant.address || 'Không có địa chỉ';
    detailModalImage.src = restaurant.image_src || 'https://placehold.co/600x400/e2e8f0/64748b?text=Không+có+ảnh';

    // (CẬP NHẬT) Cập nhật trạng thái nút "Lưu" (chỉ hiển thị nếu đã đăng nhập)
    if (currentUserId) {
        detailModalSaveBtn.classList.remove('hidden');
        updateSaveButtonUI(restaurant.name);
    } else {
        detailModalSaveBtn.classList.add('hidden');
    }

    const hasGps = restaurant.gps && restaurant.gps.includes(',');
    if (hasGps) {
        detailModalNavigateBtn.disabled = false;
        detailModalNavigateBtn.classList.remove('bg-gray-300', 'cursor-not-allowed');
        detailModalNavigateBtn.classList.add('bg-blue-500', 'hover:bg-blue-600');
    } else {
        detailModalNavigateBtn.disabled = true;
        detailModalNavigateBtn.classList.add('bg-gray-300', 'cursor-not-allowed');
        detailModalNavigateBtn.classList.remove('bg-blue-500', 'hover:bg-blue-600');
    }

    detailModal.classList.remove('hidden');
    setTimeout(() => detailModal.classList.remove('opacity-0'), 10);
}

function closeDetailModal() {
    // ... (Giữ nguyên code closeDetailModal)
    detailModal.classList.add('opacity-0');
    setTimeout(() => detailModal.classList.add('hidden'), 300);
    currentDetailRestaurant = null;
}

// (CẬP NHẬT) Giao diện nút Lưu
function updateSaveButtonUI(restaurantName) {
    // ... (Giữ nguyên code updateSaveButtonUI)
    if (savedRestaurants.includes(restaurantName)) {
        detailModalSaveIcon.innerHTML = ICON_SAVE.filled;
        detailModalSaveText.textContent = 'Đã lưu';
        detailModalSaveText.classList.add('text-red-500');
    } else {
        detailModalSaveIcon.innerHTML = ICON_SAVE.unfilled;
        detailModalSaveText.textContent = 'Lưu';
        detailModalSaveText.classList.remove('text-red-500');
    }
}

// (CẬP NHẬT) Xử lý nhấp nút "Lưu" -> GỌI API
async function handleSaveClick(e) {

    console.log("--- BẮT ĐẦU handleSaveClick ---");

    // 1. Kiểm tra Modal Data
    if (!currentDetailRestaurant) {
        console.error("LỖI: currentDetailRestaurant bị null!");
        alert("Lỗi: Không tìm thấy thông tin quán ăn. Vui lòng thử lại.");
        return;
    }
    console.log("1. Đã kiểm tra currentDetailRestaurant (OK)");

    // 2. Kiểm tra User
    if (!currentUserId) {
        console.warn("LỖI: currentUserId bị null!");
        alert("Vui lòng đăng nhập để lưu!");
        return;
    }
    console.log(`2. Đã kiểm tra currentUserId: ${currentUserId} (OK)`);
    
    const restaurantName = currentDetailRestaurant.name;
    const isSaved = savedRestaurants.includes(restaurantName);
    let endpoint = isSaved ? '/unsave' : '/save';

    console.log(`3. Quyết định Endpoint: ${endpoint} (cho quán: ${restaurantName})`);

    // 4. Cập nhật UI (tạm thời)
    if (isSaved) {
        savedRestaurants = savedRestaurants.filter(item => item !== restaurantName);
    } else {
        savedRestaurants.push(restaurantName);
    }
    updateSaveButtonUI(restaurantName);
    console.log("4. Đã cập nhật UI (tạm thời)");

    // 5. Gọi API
    try {
        console.log("5. Đang gửi request Fetch tới " + endpoint);
        const response = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                user_id: currentUserId,
                restaurant_name: restaurantName 
            })
        });

        console.log("6. Đã nhận phản hồi từ server:", response.status);

        if (!response.ok) {
            // Nếu server trả về lỗi (404, 500...)
            throw new Error(`Server báo lỗi: ${response.status}`);
        }

        const data = await response.json();
        console.log("7. Đã parse JSON:", data);

        if (!data.success) {
            // Nếu server trả về success: false (ví dụ: lỗi logic)
            throw new Error(`API báo lỗi: ${data.message}`);
        }
        
        console.log("--- KẾT THÚC handleSaveClick (THÀNH CÔNG) ---");

    } catch (err) {
        console.error("💥💥💥 LỖI NGHIÊM TRỌNG TRONG BLOC TRY...CATCH 💥💥💥", err);
        alert("Lỗi: " + err.message + ". Đang khôi phục lại trạng thái cũ.");
        
        // 8. Khôi phục lại trạng thái cũ nếu API lỗi
        if (isSaved) {
            savedRestaurants.push(restaurantName); // Thêm lại vì đã lỡ xóa ở bước 4
        } else {
            savedRestaurants = savedRestaurants.filter(item => item !== restaurantName); // Xóa đi vì đã lỡ thêm ở bước 4
        }
        updateSaveButtonUI(restaurantName);
    }
}

// Xử lý nhấp nút "Chia sẻ"
function handleShareClick() {
    // ... (Giữ nguyên code handleShareClick)
    const shareData = {
        title: currentDetailRestaurant.name,
        text: `Hãy xem thử quán ${currentDetailRestaurant.name} tại địa chỉ: ${currentDetailRestaurant.address}`,
        url: window.location.href 
    };
    try {
        if (navigator.share) {
            navigator.share(shareData);
        } else {
            navigator.clipboard.writeText(shareData.text + " " + shareData.url);
            alert('Đã sao chép link vào clipboard!');
        }
    } catch (err) {
        console.error('Lỗi khi chia sẻ:', err);
        alert('Không thể chia sẻ.');
    }
}

// Xử lý nhấp nút "Chỉ đường"
function handleNavigateFromDetail() {
    // ... (Giữ nguyên code handleNavigateFromDetail)
    const gpsString = currentDetailRestaurant.gps;
    const [destLat, destLng] = gpsString.split(',').map(c => parseFloat(c.trim()));
    
    currentDestination = { 
        name: currentDetailRestaurant.name, 
        coords: { lat: destLat, lng: destLng } 
    };
    
    closeDetailModal(); 
    openMapModal();
}

// --- CÁC HÀM XỬ LÝ MODAL BẢN ĐỒ ---
function openMapModal() {
    // ... (Giữ nguyên code openMapModal)
    mapModalTitle.textContent = `Chỉ đường tới: ${currentDestination.name}`;
    mapModal.classList.remove('hidden');
    setTimeout(() => mapModal.classList.remove('opacity-0'), 10);
    
    if (!map) {
        map = L.map('map');
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    }
    setTimeout(() => map.invalidateSize(), 200);

    renderTravelModeButtons();
    calculateAndDrawRoute();
}

function renderTravelModeButtons() {
    // ... (Giữ nguyên code renderTravelModeButtons)
    travelModeSelector.innerHTML = TRAVEL_MODES.map(mode => `
        <button class="travel-mode-btn flex items-center p-2 rounded-lg font-semibold text-gray-600 ${mode.id === currentTravelMode ? 'active' : ''}" data-mode="${mode.id}">
            ${mode.icon}
            <span class="ml-2">${mode.label}</span>
        </button>
    `).join('');

    document.querySelectorAll('.travel-mode-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            currentTravelMode = this.dataset.mode;
            renderTravelModeButtons();
            calculateAndDrawRoute();
        });
    });
}

function calculateAndDrawRoute() {
    // ... (Giữ nguyên code calculateAndDrawRoute)
    if (routingControl) map.removeControl(routingControl);
    if (animatedMarker) map.removeLayer(animatedMarker);
    startAnimationBtn.disabled = true;
    mapLoader.style.display = 'flex';
    summaryDistance.textContent = "--";
    summaryTime.textContent = "--";

    if (!userLocation) {
        alert("Vui lòng bật định vị của bạn để có thể chỉ đường!");
        mapLoader.style.display = 'none';
        closeMapModal(); 
        return;
    }

    const selectedMode = TRAVEL_MODES.find(m => m.id === currentTravelMode);
    
    routingControl = L.Routing.control({
        waypoints: [
            L.latLng(userLocation.lat, userLocation.lng),
            L.latLng(currentDestination.coords.lat, currentDestination.coords.lng)
        ],
        router: L.Routing.osrmv1({
            serviceUrl: `https://router.project-osrm.org/route/v1`,
            profile: selectedMode.profile
        }),
        addWaypoints: false,
        createMarker: () => null,
        lineOptions: { styles: [{ color: '#0d9488', opacity: 0.8, weight: 6 }] }
    }).on('routesfound', function(e) {
        mapLoader.style.display = 'none';
        const route = e.routes[0];
        const distanceInKm = route.summary.totalDistance / 1000;
        
        const speedKmh = CUSTOM_SPEEDS_KMH[currentTravelMode];
        const timeInMinutes = (distanceInKm / speedKmh) * 60;

        summaryDistance.textContent = `${distanceInKm.toFixed(2)} km`;
        summaryTime.textContent = `${Math.round(timeInMinutes)} phút`;
        startAnimationBtn.disabled = false;

        startAnimationBtn.onclick = () => {
            if (animatedMarker) map.removeLayer(animatedMarker);
            const speedMs = (speedKmh * 1000) / 3600; 

            animatedMarker = L.animatedMarker(route.coordinates, {
                distance: speedMs,
                interval: 1000,   
                icon: L.divIcon({
                    html: ICONS_SVG[currentTravelMode],
                    className: 'bg-transparent border-0',
                    iconSize: [32, 32]
                })
            });
            map.addLayer(animatedMarker);
        };
    }).addTo(map);
}

function closeMapModal() {
    // ... (Giữ nguyên code closeMapModal)
    if (animatedMarker) {
        animatedMarker.stop();
        map.removeLayer(animatedMarker);
        animatedMarker = null;
    }
    mapModal.classList.add('opacity-0');
    setTimeout(() => mapModal.classList.add('hidden'), 300);
}