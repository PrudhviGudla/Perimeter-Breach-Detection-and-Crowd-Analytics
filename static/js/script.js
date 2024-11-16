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

function sendShapeCoordinates(){
    const { scaleX, scaleY } = getScalingFactors();
    const scaledPoints = points.map(point => ({
        x: Math.round(point.x * scaleX),
        y: Math.round(point.y * scaleY)
    }));

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
    const naturalWidth = videoElement.naturalWidth || 640;  // fallback width
    const naturalHeight = videoElement.naturalHeight || 480;  // fallback height
    
    return {
        scaleX: naturalWidth / canvas.width,
        scaleY: naturalHeight / canvas.height
    };
}

function saveCurrentPerimeter() {
    if (points.length === 0) {
        alert('Please draw a perimeter first');
        return;
    }
    
    const name = prompt('Enter a name for this perimeter:');
    if (!name) return;
    
    const { scaleX, scaleY } = getScalingFactors();
    const scaledPoints = points.map(point => ({
        x: Math.round(point.x * scaleX),
        y: Math.round(point.y * scaleY)
    }));

    fetch('/save_perimeter', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name: name,
            points: scaledPoints
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert(`Perimeter "${name}" saved successfully`);
            loadPerimeters();
        } else {
            alert(`Error saving perimeter: ${data.message}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error saving perimeter. Check console for details.');
    });
}

function loadPerimeters() {
    fetch('/get_perimeters')
        .then(response => response.json())
        .then(perimeters => {
            console.log('Loaded perimeters:', perimeters);  // Debug log
            const container = document.getElementById('perimeter-select-container');
            if (perimeters.length === 0) {
                container.innerHTML = '<p>No saved perimeters</p>';
                return;
            }
            
            container.innerHTML = `
                <select id="perimeter-select" class="form-select">
                    <option value="">Select a perimeter</option>
                    ${perimeters.map(p => `
                        <option value='${JSON.stringify(p.points)}'>
                            ${p.name}
                        </option>
                    `).join('')}
                </select>
            `;
            
            document.getElementById('perimeter-select').addEventListener('change', function(e) {
                if (this.value) {
                    points = JSON.parse(this.value);
                    drawShape();
                    sendShapeCoordinates();
                }
            });
        })
        .catch(error => {
            console.error('Error loading perimeters:', error);
            const container = document.getElementById('perimeter-select-container');
            container.innerHTML = '<p>Error loading perimeters</p>';
        });
}

