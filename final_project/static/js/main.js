let video;
let cols = 3; // Number of columns
let rows = 3; // Number of rows
let threshN = 0.5; // Threshold for model N
let threshX = 0.7; // Threshold for model X
let alertDiv;
let hoveredZone = -1; // Variable to store the index of the hovered zone
let detectedZones = []; // Variable to store the detected fire zones

function setup() {
  let canvas = createCanvas(640, 480);
  canvas.parent('video-container');
  video = createCapture(VIDEO);
  video.size(width, height);
  video.hide(); // Hide the original video element
  alertDiv = select('#alert');

  // Add event listeners to update grid and thresholds
  document.getElementById('numCols').addEventListener('input', updateGridSettings);
  document.getElementById('numRows').addEventListener('input', updateGridSettings);
  document.getElementById('threshN').addEventListener('input', updateThresholds);
  document.getElementById('threshX').addEventListener('input', updateThresholds);

  setTimeout(captureFrame, 1000); // Start capturing frames after a delay
}

function draw() {
  background(0);
  image(video, 0, 0, width, height);
  drawGrid(cols, rows);
}

function drawGrid(cols, rows) {
  let colWidth = width / cols; // Width of each column
  let rowHeight = height / rows; // Height of each row

  // Draw grid and highlight hovered or detected zone
  for (let col = 0; col < cols; col++) {
    for (let row = 0; row < rows; row++) {
      let x = col * colWidth;
      let y = row * rowHeight;
      let zoneIndex = col + row * cols;

      // Check if this zone is in the detected zones
      let detected = detectedZones.includes(zoneIndex);

      if (hoveredZone === zoneIndex || detected) {
        fill(255, 0, 0, 100); // Highlight color for detected zone
        rect(x, y, colWidth, rowHeight);

        // Display zone number
        fill(0);
        textAlign(CENTER, CENTER);
        textSize(32);
        text(zoneIndex + 1, x + colWidth / 2, y + rowHeight / 2);
      }

      stroke(48, 90, 87, 255); // Grid line color
      strokeWeight(1); // Grid line thickness
      noFill();
      rect(x, y, colWidth, rowHeight);
    }
  }
}

function captureFrame() {
  video.loadPixels();
  if (video.pixels.length > 0) {
    const canvas = createGraphics(video.width, video.height);
    canvas.image(video, 0, 0, video.width, video.height);
    canvas.loadPixels();
    const frameData = canvas.elt.toDataURL('image/jpeg').split(',')[1];

    fetch('/detect', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        frame: frameData,
        width: canvas.width,
        height: canvas.height,
        num_rows: rows,
        num_cols: cols,
        thresh_n: threshN,
        thresh_x: threshX
      })
    })
    .then(response => response.json())
    .then(data => {
      detectedZones = data.detected_zones.map(zone => zone.index); // Update detected zones with their indexes
      if (detectedZones.length > 0) {
        let zoneMessages = data.detected_zones.map((zone) => `Fire detected in zone ${zone.index + 1} with confidence: ${zone.confidence}`).join('<br>');
        alertDiv.style('display', 'block');
        alertDiv.html(zoneMessages);

      } else {
        alertDiv.addClass('fade-out');
        setTimeout(() => {
          alertDiv.style('display', 'none');
          alertDiv.removeClass('fade-out');
        }, 2000); // Match this duration to the CSS transition time
      }
      setTimeout(captureFrame, 1000);  // Adjust the interval as needed
    })
    .catch(error => {
      console.error('Error detecting fire: ', error);
      setTimeout(captureFrame, 1000);  // Retry after an error
    });
  } else {
    setTimeout(captureFrame, 1000);  // Retry if pixels are not loaded
  }
}

function updateGridSettings() {
  cols = parseInt(document.getElementById('numCols').value);
  rows = parseInt(document.getElementById('numRows').value);
  console.log(`Updated grid settings: ${cols} columns, ${rows} rows`); // Debugging line
}

function updateThresholds() {
  threshN = parseFloat(document.getElementById('threshN').value);
  threshX = parseFloat(document.getElementById('threshX').value);
  console.log(`Updated thresholds: threshN=${threshN}, threshX=${threshX}`); // Debugging line
}

function mouseMoved() {
  let colWidth = width / cols; // Width of each column
  let rowHeight = height / rows; // Height of each row
  let col = Math.floor(mouseX / colWidth);
  let row = Math.floor(mouseY / rowHeight);

  if (col >= 0 && col < cols && row >= 0 && row < rows) {
    hoveredZone = col + row * cols;
  } else {
    hoveredZone = -1;
  }

  return false; // Prevent default
}
