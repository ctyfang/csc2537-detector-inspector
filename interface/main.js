let currentSceneIndex = 1;
const totalScenes = 10; // Adjust this to match how many scenes you have

const visualizationSelect = document.getElementById("visualization-options");
const sceneLabel = document.getElementById("scene-label");

let lastX = 0;
let lastY = 0;

let alphaX = 50.0 / 155.0;
let alphaY = -50.0 / 185.0;
let betaX = -205.0 * 50.0 / 155.0;
let betaY = 535.0 * 50.0 / 185.0;

let radius = 10;
let imageRadius = 50;

let yCamFixed = 2;
let intrinsics = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// Set up the images
const bevCanvas = document.getElementById('bev-img');
const bevCtx = bevCanvas.getContext('2d');
const bevImage = new Image();
const imageCanvas = document.getElementById('image-img');
const imageCtx = imageCanvas.getContext('2d');
const imageView = new Image();
// updateImages();

function padSceneIndex(index) {
    return index.toString().padStart(3, '0'); // scene_001
}

function getSceneFolder(index) {
    return `scenes/scene_${padSceneIndex(index)}`;
}

function computeImageCoordinates() {
    let xCam = alphaX * lastX + betaX;
    let zCam = alphaY * lastY + betaY;
    return {
        'x': (xCam * intrinsics[0][0] + zCam * intrinsics[0][2]) / zCam,
        'y': (yCamFixed * intrinsics[1][1] + zCam * intrinsics[1][2]) / zCam
    }
}

function updateCalibration() {
    // The name of your CSV file
    const scenePath = getSceneFolder(currentSceneIndex);
    const csvFileName = `${scenePath}/intrinsics.csv`;

    // Use fetch API to get the CSV file from the same directory
    fetch(csvFileName)
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    })
    .then(csvContent => {
        // Parse CSV content using PapaParse
        const results = Papa.parse(csvContent, {
            dynamicTyping: true, // Convert numeric strings to numbers
            skipEmptyLines: true // Skip empty lines
        });
        
        intrinsics = results.data;
        console.log("Updated intrinsics")
    })
    .catch(error => {
        document.getElementById('output').innerHTML = 
            `<p>Error loading the CSV file: ${error.message}</p>
             <p>Make sure "${csvFileName}" is in the same folder as this HTML file.</p>`;
    });
}
function updateImages() {
    const selectedMode = visualizationSelect.value;
    const scenePath = getSceneFolder(currentSceneIndex);
    updateCalibration();

    const imageMap = {
        'overlapping': ['overlapping_bev.png', 'overlapping_image.png'],
        'boxes-data': ['overlapping_bev.png', 'overlapping_image.png'],
        'highlight-fn': ['false_negative_bev.png', 'false_negative_image.png'],
        'highlight-fp': ['false_positive_bev.png', 'false_positive_image.png']
    };

    const [bevFile, imageFile] = imageMap[selectedMode];

    bevImage.src = `${scenePath}/${bevFile}`;
    imageView.src = `${scenePath}/${imageFile}`;
    sceneLabel.textContent = `Scene ${padSceneIndex(currentSceneIndex)}`;

    // Draw the image
    bevCtx.drawImage(bevImage, 0, 0, bevCanvas.width, bevCanvas.height);
    imageCtx.drawImage(imageView, 0, 0, imageCanvas.width, imageCanvas.height);

    bevCtx.beginPath();
    bevCtx.arc(lastX, lastY, radius, 0, Math.PI * 2);
    bevCtx.strokeStyle = '#000';
    bevCtx.lineWidth = 2;
    bevCtx.stroke();

    imageCoords = computeImageCoordinates();
    imageCtx.beginPath();
    imageCtx.arc(imageCoords.x, imageCoords.y, imageRadius, 0, Math.PI * 2);
    imageCtx.strokeStyle = '#000';
    imageCtx.lineWidth = 2;
    imageCtx.stroke();
}

// Event Listeners
document.getElementById("prev-scene").addEventListener("click", () => {
    currentSceneIndex = (currentSceneIndex - 1 + totalScenes) % totalScenes;
    updateImages();
});

document.getElementById("next-scene").addEventListener("click", () => {
    currentSceneIndex = (currentSceneIndex + 1) % totalScenes;
    updateImages();
});

visualizationSelect.addEventListener("change", updateImages);

// Initialize on page load
document.addEventListener("DOMContentLoaded", updateImages);

// Add click event listener
bevCanvas.addEventListener('click', function(event) {
    // Get position relative to imageCanvas
    const rect = bevCanvas.getBoundingClientRect();
    const scaleX = bevCanvas.width / rect.width;
    const scaleY = bevCanvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    // Update coordinates display
    // coordinatesDisplay.textContent = `Coordinates: (${Math.round(x)}, ${Math.round(y)})`;
    
    lastX = x;
    lastY = y;
    
    // Redraw canvas with new circle
    updateImages();
});
