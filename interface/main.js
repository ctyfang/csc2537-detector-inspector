let currentSceneIndex = 1;
const totalScenes = 10; // Adjust this to match how many scenes you have

const visualizationSelect = document.getElementById("visualization-options");
const sceneLabel = document.getElementById("scene-label");

// Set up the images
const bevCanvas = document.getElementById('bev-img');
const bevCtx = bevCanvas.getContext('2d');
const bevImage = new Image();
const imageCanvas = document.getElementById('image-img');
const imageCtx = imageCanvas.getContext('2d');
const imageView = new Image();

function padSceneIndex(index) {
    return index.toString().padStart(3, '0'); // scene_001
}

function getSceneFolder(index) {
    return `scenes/scene_${padSceneIndex(index)}`;
}

function updateImages() {
    const selectedMode = visualizationSelect.value;
    const scenePath = getSceneFolder(currentSceneIndex);

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
