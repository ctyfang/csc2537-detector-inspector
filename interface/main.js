// Main JavaScript file for Bird's Eye View Bounding Box Visualization

// Global variables
let boundingBoxes = [];
let scale = 1;
let clickPoint = null;
let canvas, ctx;
let backgroundImage = null;
let imageProperties = {
    width: 0,
    height: 0,
    aspectRatio: 1,
    loaded: false
};


// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    canvas = document.getElementById('image_canvas');
    ctx = canvas.getContext('2d');

    // Set up event listeners
    canvas.addEventListener('click', handleCanvasClick);
    
    loadLocalImage();
    
    // Wait a moment for the image to load before loading CSV
    setTimeout(() => {
        loadLocalCSV();
    }, 500);
});

// Handle canvas click events
function handleCanvasClick(event) {
    const rect = canvas.getBoundingClientRect();
    
    // Calculate coordinates in the image space
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;
    
    // Convert canvas coordinates to image coordinates
    const x = canvasX / canvas.width * imageProperties.width;
    const y = canvasY / canvas.height * imageProperties.height;
    
    // Log the clicked coordinates to the console
    console.log(`Clicked at image coordinates: (${x.toFixed(2)}, ${y.toFixed(2)})`);
    
    // Update the coordinates display
    const coordinatesDiv = document.getElementById('coordinates');
    coordinatesDiv.innerHTML = `Clicked at: (${x.toFixed(2)}, ${y.toFixed(2)})`;
    
    // Update clickPoint and render
    clickPoint = { x, y };
    renderVisualization();
}

// Parse CSV data
function parseCSV(csvData) {
    Papa.parse(csvData, {
        header: false, // CSV doesn't have headers but specific column order
        dynamicTyping: true, // Automatically convert numeric strings to numbers
        skipEmptyLines: true,
        complete: function(results) {
            console.log('Parsed CSV data:', results);
            
            if (results.data && results.data.length > 0) {
                // Process the custom CSV format:
                // [class, x1, y1, x2, y2, x, y, z, length, width, height, yaw]
                boundingBoxes = results.data.map((row, index) => {
                    // Ensure the row has enough columns
                    if (row.length < 12) {
                        console.error(`Row ${index} has fewer than 12 columns:`, row);
                        return null;
                    }
                    
                    // Extract 2D bounding box from (x1,y1,x2,y2)
                    const className = row[0] || `Box-${index}`;
                    const x1 = parseFloat(row[1]);
                    const y1 = parseFloat(row[2]);
                    const x2 = parseFloat(row[3]);
                    const y2 = parseFloat(row[4]);
                    
                    // Calculate width and height from corner points
                    const x = Math.min(x1, x2);
                    const y = Math.min(y1, y2);
                    const width = Math.abs(x2 - x1);
                    const height = Math.abs(y2 - y1);
                    
                    // Store 3D data too
                    const box3D = {
                        x: parseFloat(row[5]),
                        y: parseFloat(row[6]),
                        z: parseFloat(row[7]),
                        length: parseFloat(row[8]),
                        width: parseFloat(row[9]),
                        height: parseFloat(row[10]),
                        yaw: parseFloat(row[11])
                    };
                    
                    return {
                        label: className,
                        x: x,
                        y: y,
                        width: width,
                        height: height,
                        x1: x1,
                        y1: y1,
                        x2: x2,
                        y2: y2,
                        box3D: box3D
                    };
                }).filter(box => box !== null && box.width > 0 && box.height > 0); // Filter out invalid boxes
                
                console.log('Processed bounding boxes:', boundingBoxes);
                
                // Validate data format
                if (boundingBoxes.length === 0) {
                    alert('No valid bounding boxes found in the CSV. Please check your file format.');
                    return;
                }
                
                // Update visualization and table
                renderVisualization();
                updateBoundingBoxTable();
            } else {
                alert('No data found in the CSV file.');
            }
        },
        error: function(error) {
            console.error('Error parsing CSV:', error);
            alert('Error parsing CSV: ' + error.message);
        }
    });
}

// Validate bounding box data format
function validateBoundingBoxData(data) {
    return true;
}

// Load a local image from the same folder
function loadLocalImage() {
    var filename = "../data/images/f1275002-842e-3571-8f7d-05816bc7cf56_315977575849927210.jpg"
    backgroundImage = new Image();
    
    backgroundImage.onload = function() {
        imageProperties.width = backgroundImage.width;
        imageProperties.height = backgroundImage.height;
        imageProperties.aspectRatio = backgroundImage.width / backgroundImage.height;
        imageProperties.loaded = true;
        
        // Adjust canvas size to match image proportions while maintaining a reasonable size
        const maxWidth = 1000;
        const maxHeight = 800;
        
        if (backgroundImage.width > maxWidth || backgroundImage.height > maxHeight) {
            if (backgroundImage.width / maxWidth > backgroundImage.height / maxHeight) {
                canvas.width = maxWidth;
                canvas.height = maxWidth / imageProperties.aspectRatio;
            } else {
                canvas.height = maxHeight;
                canvas.width = maxHeight * imageProperties.aspectRatio;
            }
        } else {
            canvas.width = backgroundImage.width;
            canvas.height = backgroundImage.height;
        }
        
        // Update offset to center the coordinate system
        offset.x = 0;
        offset.y = 0;
        
        console.log(`Image loaded: ${filename} (${backgroundImage.width}x${backgroundImage.height})`);
        renderVisualization();
    };
    
    backgroundImage.onerror = function() {
        console.error(`Error loading image: ${filename}`);
        alert(`Could not load image: ${filename}. Make sure the file exists in the same folder as index.html.`);
        imageProperties.loaded = false;
    };
    
    // Load the image (must be served from a web server due to browser security restrictions)
    backgroundImage.src = filename;
}

// Load a CSV file from the same directory
function loadLocalCSV() {
    const filename = '../data/labels/f1275002-842e-3571-8f7d-05816bc7cf56_315977575849927210.txt'; // Change this to your CSV filename
    
    fetch(filename)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Could not find ${filename}. Make sure it's in the same directory as index.html.`);
            }
            return response.text();
        })
        .then(csvData => {
            console.log('Successfully loaded CSV file:', filename);
            parseCSV(csvData);
        })
        .catch(error => {
            console.error('Error loading CSV file:', error);
            alert(`Error loading CSV file: ${error.message}`);
        });
}

// Update the bounding box table
function updateBoundingBoxTable() {
    const tableBody = document.getElementById('boxesTableBody');
    tableBody.innerHTML = '';
    
    boundingBoxes.forEach((box, index) => {
        const row = document.createElement('tr');
        
        // Add basic 2D info
        const classCell = document.createElement('td');
        classCell.textContent = box.label;
        
        const cornerCell = document.createElement('td');
        cornerCell.textContent = `(${box.x1.toFixed(1)}, ${box.y1.toFixed(1)}) to (${box.x2.toFixed(1)}, ${box.y2.toFixed(1)})`;
        
        const dimCell = document.createElement('td');
        dimCell.textContent = `${box.width.toFixed(1)} × ${box.height.toFixed(1)}`;
        
        // Add 3D info if available
        const pos3dCell = document.createElement('td');
        if (box.box3D) {
            pos3dCell.textContent = `(${box.box3D.x.toFixed(1)}, ${box.box3D.y.toFixed(1)}, ${box.box3D.z.toFixed(1)})`;
        } else {
            pos3dCell.textContent = 'N/A';
        }
        
        const dim3dCell = document.createElement('td');
        if (box.box3D) {
            dim3dCell.textContent = `${box.box3D.length.toFixed(1)} × ${box.box3D.width.toFixed(1)} × ${box.box3D.height.toFixed(1)}`;
        } else {
            dim3dCell.textContent = 'N/A';
        }
        
        const yawCell = document.createElement('td');
        if (box.box3D) {
            yawCell.textContent = `${box.box3D.yaw.toFixed(2)} rad`;
        } else {
            yawCell.textContent = 'N/A';
        }
        
        row.appendChild(classCell);
        row.appendChild(cornerCell);
        row.appendChild(dimCell);
        row.appendChild(pos3dCell);
        row.appendChild(dim3dCell);
        row.appendChild(yawCell);
        
        // Highlight row on hover to see relationship with visualization
        row.addEventListener('mouseover', () => {
            // Could add highlight functionality in the future
        });
        
        tableBody.appendChild(row);
    });
}

function drawAxisLabels() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.font = '12px Arial';
    
    // Origin indicator
    ctx.fillText('(0,0)', 5, 15);
    
    // X-axis direction
    ctx.fillText('X →', canvas.width - 30, 15);
    
    // Y-axis direction
    ctx.fillText('Y ↓', 5, canvas.height - 10);
}// Main JavaScript file for Bird's Eye View Bounding Box Visualization

// Render the visualization
function renderVisualization() {
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background image if loaded
    if (imageProperties.loaded && backgroundImage) {
        // Draw the image scaled to fit the canvas
        ctx.drawImage(
            backgroundImage, 
            0, 
            0, 
            canvas.width, 
            canvas.height
        );
    } else {
        // Use plain background color if image not loaded
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Display message to load image
        ctx.fillStyle = '#999';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Loading image...', canvas.width/2, canvas.height/2);
        ctx.textAlign = 'left';
    }
    
    // Draw coordinate axes labels
    drawAxisLabels();
    
    // Draw bounding boxes
    drawBoundingBoxes();
    
    // Draw click point if exists
    if (clickPoint) {
        drawClickPoint();
    }
}

// Draw grid lines
function drawGrid() {
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 0.5;
    
    const gridSize = 20 * scale;
    
    // Draw vertical grid lines
    for (let i = 0; i < canvas.width; i += gridSize) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
    }
    
    // Draw horizontal grid lines
    for (let i = 0; i < canvas.height; i += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
    }
    
    // Draw origin and axes
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(canvas.width, 0);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, canvas.height);
    ctx.stroke();
}

// Draw coordinate axes
function drawAxes() {
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(0, offset.y * scale);
    ctx.lineTo(canvas.width, offset.y * scale);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(offset.x * scale, 0);
    ctx.lineTo(offset.x * scale, canvas.height);
    ctx.stroke();
    
    // Add labels
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.fillText('X', canvas.width - 15, offset.y * scale - 5);
    ctx.fillText('Y', offset.x * scale + 5, 15);
    ctx.fillText('(0,0)', offset.x * scale - 25, offset.y * scale + 15);
    
    // Add scale markers
    const markerInterval = 20;
    const markerSize = 5;
    
    // X-axis markers
    for (let i = markerInterval; i < canvas.width / scale; i += markerInterval) {
        const x = (i + offset.x) * scale;
        ctx.beginPath();
        ctx.moveTo(x, offset.y * scale - markerSize/2);
        ctx.lineTo(x, offset.y * scale + markerSize/2);
        ctx.stroke();
        
        if (i % (markerInterval * 5) === 0) {
            ctx.fillText(i, x - 5, offset.y * scale + 20);
        }
    }
    
    // Y-axis markers
    for (let i = markerInterval; i < canvas.height / scale; i += markerInterval) {
        const y = (i + offset.y) * scale;
        ctx.beginPath();
        ctx.moveTo(offset.x * scale - markerSize/2, y);
        ctx.lineTo(offset.x * scale + markerSize/2, y);
        ctx.stroke();
        
        if (i % (markerInterval * 5) === 0) {
            ctx.fillText(i, offset.x * scale - 25, y + 5);
        }
    }
}

// Draw bounding boxes
function drawBoundingBoxes() {
    boundingBoxes.forEach((box, index) => {
        const color = '#008000';
        
        ctx.strokeStyle = color;
        ctx.fillStyle = color + (imageProperties.loaded ? '22' : '11'); // More opacity if image is loaded
        ctx.lineWidth = imageProperties.loaded ? 3 : 2; // Thicker lines if image is loaded
        
        // Use direct coordinates (origin at top-left)
        let x = box.x;
        let y = box.y;
        let width = box.width;
        let height = box.height;

        ctx.fillRect(x, y, width, height);
        ctx.strokeRect(x, y, width, height);
    });
}

// Draw click point if exists
function drawClickPoint() {
    ctx.fillStyle = '#FF0000';
    ctx.beginPath();
    ctx.arc(clickPoint.x * scale, clickPoint.y * scale, 5, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw guidelines
    ctx.strokeStyle = '#FF0000';
    ctx.setLineDash([5, 3]);
    
    // Horizontal guideline
    ctx.beginPath();
    ctx.moveTo(0, clickPoint.y * scale);
    ctx.lineTo(canvas.width, clickPoint.y * scale);
    ctx.stroke();
    
    // Vertical guideline
    ctx.beginPath();
    ctx.moveTo(clickPoint.x * scale, 0);
    ctx.lineTo(clickPoint.x * scale, canvas.height);
    ctx.stroke();
    
    // Reset line dash
    ctx.setLineDash([]);
}