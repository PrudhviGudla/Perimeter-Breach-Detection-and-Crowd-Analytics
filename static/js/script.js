let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let drawing = false;
let startX, startY, endX, endY;
let setLineMode = false;
let buzzerOn = false;
let analytics = {
    peopleCount: 0,
    crowdDensity: 'Low',
    breachCount: 0,
    unusualBehavior: true,
    behaviorDetails: ["None"]
};
let points = [];
let isDrawingShape = false;
let currentPerimeterName = '';

function initCanvas() {
    const videoContainer = document.getElementById('video-container');
    const videoElement = document.getElementById('video-stream');
    
    // Set canvas size to match video container
    canvas.width = videoContainer.offsetWidth;
    canvas.height = videoContainer.offsetHeight;
    
    // Ensure video is visible
    videoElement.style.display = 'block';
    
    // Setup metadata handlers
    setupVideoMetadataHandler();
    
    // Redraw shape if exists
    if (points.length > 0) {
        drawShape();
    }
}

window.addEventListener('load', function() {
    initCanvas();
    loadPerimeters();
});

window.addEventListener('resize', initCanvas);

document.getElementById('set-line').addEventListener('click', function(){
    setLineMode = true;
    points = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    alert('Click to add points. Double click to complete the shape. You can save the perimeter after drawing.');
});

canvas.addEventListener('mousedown', function(e){
    if(setLineMode && !isDrawingShape){
        isDrawingShape = true;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        points.push({x, y});
        drawShape();
    }
});

canvas.addEventListener('click', function(e){
    if(setLineMode && isDrawingShape){
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        points.push({x, y});
        drawShape();
    }
});

canvas.addEventListener('dblclick', function(e){
    if(setLineMode && isDrawingShape){
        isDrawingShape = false;
        setLineMode = false;
        drawShape();
        sendShapeCoordinates();
    }
});

function drawShape() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (points.length === 0) return;
    
    // Make sure canvas is transparent
    ctx.globalAlpha = 0.8;
    
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    
    if (!isDrawingShape) {
        ctx.closePath();
    }
    
    // Draw stroke
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.stroke();
    
    // Draw fill
    if (!isDrawingShape) {
        ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
        ctx.fill();
    }

    // Draw points
    ctx.globalAlpha = 1.0;
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
    });
}

function sendShapeCoordinates() {
    const { scaleX, scaleY } = getScalingFactors();
    
    // Get video element to check actual display dimensions
    const videoElement = document.getElementById('video-stream');
    const videoContainer = document.getElementById('video-container');
    
    // Calculate the actual display size of the video within the container
    let displayWidth, displayHeight;
    const videoAspect = (videoElement.videoWidth || videoElement.naturalWidth) / 
                        (videoElement.videoHeight || videoElement.naturalHeight);
    const containerAspect = videoContainer.offsetWidth / videoContainer.offsetHeight;
    
    if (videoAspect > containerAspect) {
        // Video is wider than container (letterboxing)
        displayWidth = videoContainer.offsetWidth;
        displayHeight = displayWidth / videoAspect;
    } else {
        // Video is taller than container (pillarboxing)
        displayHeight = videoContainer.offsetHeight;
        displayWidth = displayHeight * videoAspect;
    }
    
    // Calculate offsets for letterboxing/pillarboxing
    const xOffset = (videoContainer.offsetWidth - displayWidth) / 2;
    const yOffset = (videoContainer.offsetHeight - displayHeight) / 2;
    
    const scaledPoints = points.map(point => {
        // Adjust for letterboxing/pillarboxing
        const adjustedX = point.x - xOffset;
        const adjustedY = point.y - yOffset;
        
        // Scale to actual video dimensions
        return {
            x: Math.round(adjustedX * (videoElement.videoWidth || videoElement.naturalWidth) / displayWidth),
            y: Math.round(adjustedY * (videoElement.videoHeight || videoElement.naturalHeight) / displayHeight)
        };
    });

    fetch('/set_shape', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            points: scaledPoints
        })
    });
}

function updateBuzzerButton(isOn) {
    const button = document.getElementById('toggle-buzzer');
    buzzerOn = isOn;
    button.textContent = buzzerOn ? 'Turn Off Buzzer' : 'Turn On Buzzer';
}

function handleBreach() {
    updateBuzzerButton(true);
}

document.getElementById('toggle-buzzer')
    .addEventListener('click', function(){
    fetch('/toggle_buzzer', {
        method: 'POST'
    }).then(response => response.json())
      .then(data => {
        updateBuzzerButton(data.buzzer_on);
    });
});

function checkBuzzerState() {
    fetch('/get_buzzer_state')
        .then(response => response.json())
        .then(data => {
            updateBuzzerButton(data.buzzer_on);
        });
}

function updateAnalytics() {
    fetch('/get_analytics')
        .then(response => response.json())
        .then(data => {
            analytics = data;
            displayAnalytics();
        });
}

function displayAnalytics() {
    const statsContainer = document.getElementById('analytics-container');
    statsContainer.innerHTML = `
        <div class="stat-box">
            <h3>People Count</h3>
            <p>${analytics.peopleCount}</p>
        </div>
        <div class="stat-box">
            <h3>Crowd Density</h3>
            <p>${analytics.crowdDensity}</p>
        </div>
        <div class="stat-box">
            <h3>Breach Count</h3>
            <p>${analytics.breachCount}</p>
        </div>
        <div class="stat-box">
            <h3>Unusual Behavior</h3>
            <p>${analytics.behaviorDetails.join(', ')}</p>
        </div>
    `;
}

setInterval(() => {
    checkBuzzerState();
    updateAnalytics();
}, 2000);

function getScalingFactors() {
    const videoElement = document.getElementById('video-stream');
    const videoContainer = document.getElementById('video-container');
    
    // Get the intrinsic dimensions of the video stream
    let naturalWidth, naturalHeight;
    
    if (videoElement.complete || videoElement.naturalWidth) {
        // For image elements that have loaded
        naturalWidth = videoElement.naturalWidth;
        naturalHeight = videoElement.naturalHeight;
    } else if (videoElement.videoWidth) {
        // For video elements
        naturalWidth = videoElement.videoWidth;
        naturalHeight = videoElement.videoHeight;
    } else {
        // Fallback to container dimensions if actual dimensions aren't available
        naturalWidth = videoContainer.offsetWidth;
        naturalHeight = videoContainer.offsetHeight;
        console.warn('Using container dimensions as fallback. This may cause scaling issues.');
    }
    
    console.log(`Video natural dimensions: ${naturalWidth}x${naturalHeight}`);
    console.log(`Canvas dimensions: ${canvas.width}x${canvas.height}`);
    
    // Calculate aspect ratios to handle letterboxing/pillarboxing
    const videoAspect = naturalWidth / naturalHeight;
    const canvasAspect = canvas.width / canvas.height;
    
    let scaleX, scaleY;
    
    if (videoAspect > canvasAspect) {
        // Video is wider than canvas (letterboxing)
        const displayHeight = canvas.width / videoAspect;
        const yOffset = (canvas.height - displayHeight) / 2;
        
        scaleX = naturalWidth / canvas.width;
        scaleY = naturalHeight / displayHeight;
    } else {
        // Video is taller than canvas (pillarboxing)
        const displayWidth = canvas.height * videoAspect;
        const xOffset = (canvas.width - displayWidth) / 2;
        
        scaleX = naturalWidth / displayWidth;
        scaleY = naturalHeight / canvas.height;
    }
    
    console.log(`Scaling factors: X=${scaleX}, Y=${scaleY}`);
    return { scaleX, scaleY };
}

function setupVideoMetadataHandler() {
    const videoElement = document.getElementById('video-stream');
    
    // For image elements
    videoElement.onload = function() {
        console.log('Image loaded, dimensions available');
        // Redraw any existing shapes with correct scaling
        if (points.length > 0) {
            drawShape();
        }
    };
    
    // For video elements (in case the stream is treated as video)
    videoElement.onloadedmetadata = function() {
        console.log('Video metadata loaded, dimensions available');
        // Redraw any existing shapes with correct scaling
        if (points.length > 0) {
            drawShape();
        }
    };
}

// Add this function (it's missing from your current script.js)
async function saveCurrentPerimeter() {
    if (points.length === 0) {
        alert('No perimeter drawn to save!');
        return;
    }
    
    const name = prompt('Enter perimeter name:');
    if (!name) return;
    
    try {
        const { scaleX, scaleY } = getScalingFactors();
        const videoElement = document.getElementById('video-stream');
        const videoContainer = document.getElementById('video-container');
        
        // Calculate the actual display size of the video within the container
        let displayWidth, displayHeight;
        const videoAspect = (videoElement.videoWidth || videoElement.naturalWidth) / (videoElement.videoHeight || videoElement.naturalHeight);
        const containerAspect = videoContainer.offsetWidth / videoContainer.offsetHeight;
        
        if (videoAspect > containerAspect) {
            displayWidth = videoContainer.offsetWidth;
            displayHeight = displayWidth / videoAspect;
        } else {
            displayHeight = videoContainer.offsetHeight;
            displayWidth = displayHeight * videoAspect;
        }
        
        const xOffset = (videoContainer.offsetWidth - displayWidth) / 2;
        const yOffset = (videoContainer.offsetHeight - displayHeight) / 2;
        
        const scaledPoints = points.map(point => {
            const adjustedX = point.x - xOffset;
            const adjustedY = point.y - yOffset;
            return {
                x: Math.round(adjustedX * (videoElement.videoWidth || videoElement.naturalWidth) / displayWidth),
                y: Math.round(adjustedY * (videoElement.videoHeight || videoElement.naturalHeight) / displayHeight)
            };
        });
        
        const response = await fetch('/save_perimeter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                points: scaledPoints
            })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            alert('Perimeter saved successfully!');
            loadPerimeters(); // Refresh the dropdown
        } else {
            alert('Error saving perimeter: ' + data.message);
        }
    } catch (error) {
        console.error('Error saving perimeter:', error);
        alert('Error saving perimeter!');
    }
}


function loadPerimeters() {
    fetch('/get_perimeters')
    .then(response => response.json())
    .then(data => {
        const container = document.getElementById('perimeter-select-container');
        
        // Check if the response has the expected structure
        if (data.status === 'success' && data.perimeters && data.perimeters.length > 0) {
            container.innerHTML = `
                <select id="perimeter-select" class="form-select">
                    <option value="">Select a saved perimeter</option>
                    ${data.perimeters.map(p => 
                        `<option value='${JSON.stringify(p.points)}' data-id='${p.id}'>${p.name}</option>`
                    ).join('')}
                </select>
                <button class="btn" onclick="deleteSelectedPerimeter()">Delete Selected</button>
            `;
        } else if (data.status === 'error') {
            container.innerHTML = `<div class="error">Error: ${data.message}</div>`;
        } else {
            container.innerHTML = '<div class="info">No saved perimeters</div>';
        }
        
        // Add event listener for perimeter selection
        const selectElement = document.getElementById('perimeter-select');
        if (selectElement) {
            selectElement.addEventListener('change', function(e) {
                if (this.value) {
                    const perimeterPoints = JSON.parse(this.value);
                    loadPerimeterPoints(perimeterPoints);
                }
            });
        }
    })
    .catch(error => {
        console.error('Error loading perimeters:', error);
        const container = document.getElementById('perimeter-select-container');
        container.innerHTML = '<div class="error">Error loading perimeters</div>';
    });
}

// Add this helper function for loading perimeter points
function loadPerimeterPoints(perimeterPoints) {
    const videoElement = document.getElementById('video-stream');
    const videoContainer = document.getElementById('video-container');
    
    // Calculate the actual display size of the video within the container
    let displayWidth, displayHeight;
    const videoAspect = (videoElement.videoWidth || videoElement.naturalWidth) / (videoElement.videoHeight || videoElement.naturalHeight);
    const containerAspect = videoContainer.offsetWidth / videoContainer.offsetHeight;
    
    if (videoAspect > containerAspect) {
        displayWidth = videoContainer.offsetWidth;
        displayHeight = displayWidth / videoAspect;
    } else {
        displayHeight = videoContainer.offsetHeight;
        displayWidth = displayHeight * videoAspect;
    }
    
    const xOffset = (videoContainer.offsetWidth - displayWidth) / 2;
    const yOffset = (videoContainer.offsetHeight - displayHeight) / 2;
    
    // Convert backend coordinates to canvas coordinates
    points = perimeterPoints.map(point => {
        const scaledX = point.x * displayWidth / (videoElement.videoWidth || videoElement.naturalWidth);
        const scaledY = point.y * displayHeight / (videoElement.videoHeight || videoElement.naturalHeight);
        return {
            x: Math.round(scaledX + xOffset),
            y: Math.round(scaledY + yOffset)
        };
    });
    
    drawShape();
    sendShapeCoordinates();
}

async function deleteSelectedPerimeter() {
    const selectElement = document.getElementById('perimeter-select');
    if (!selectElement || !selectElement.value) {
        alert('Please select a perimeter to delete');
        return;
    }
    
    const selectedOption = selectElement.options[selectElement.selectedIndex];
    const perimeterId = selectedOption.getAttribute('data-id'); // We need to modify the option creation
    
    if (!perimeterId) {
        alert('Unable to delete perimeter');
        return;
    }
    
    if (!confirm(`Are you sure you want to delete "${selectedOption.text}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/delete_perimeter/${perimeterId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            alert('Perimeter deleted successfully!');
            points = []; // Clear current perimeter
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            loadPerimeters(); // Refresh the dropdown
        } else {
            alert('Error deleting perimeter: ' + data.message);
        }
    } catch (error) {
        console.error('Error deleting perimeter:', error);
        alert('Error deleting perimeter!');
    }
}



function debugVideoDimensions() {
    const videoElement = document.getElementById('video-stream');
    const videoContainer = document.getElementById('video-container');
    
    console.log('Video element:', videoElement);
    console.log('Natural dimensions:', videoElement.naturalWidth, 'x', videoElement.naturalHeight);
    console.log('Video dimensions:', videoElement.videoWidth, 'x', videoElement.videoHeight);
    console.log('Container dimensions:', videoContainer.offsetWidth, 'x', videoContainer.offsetHeight);
    console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
}

setInterval(debugVideoDimensions, 5000);

