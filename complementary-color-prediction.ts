/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// tslint:disable-next-line:max-line-length
import {Array1D, CostReduction, FeedEntry, Graph, InCPUMemoryShuffledInputProviderBuilder, NDArrayMath, NDArrayMathGPU, Session, MomentumOptimizer, Tensor} from 'deeplearn';
declare var d3:any;

class ComplementaryColorModel {
  // Runs training.
  session: Session;

  // Encapsulates math operations on the CPU and GPU.
  math: NDArrayMath = new NDArrayMathGPU();

  // An optimizer with a certain initial learning rate. Used for training.
  initialLearningRate = 0.042;
  optimizer: MomentumOptimizer;
  // Momentum parameter for MomentumOptimizer, usually set to ~0.9
  momentum = 0.9;

  // Each training batch will be on this many examples.
  batchSize = 300;

  inputTensor: Tensor;
  targetTensor: Tensor;
  costTensor: Tensor;
  predictionTensor: Tensor;

  // Maps tensors to InputProviders.
  feedEntries: FeedEntry[];

  constructor() {
    this.optimizer = new MomentumOptimizer(this.initialLearningRate, this.momentum)
  }

  /**
   * Constructs the graph of the model. Call this method before training.
   */
  setupSession(): void {
    const graph = new Graph();

    // This tensor contains the input. In this case, it is a scalar.
    this.inputTensor = graph.placeholder('input RGB value', [3]);

    // This tensor contains the target.
    this.targetTensor = graph.placeholder('output RGB value', [3]);
/**/
    // Create 3 fully connected layers, each with half the number of nodes of
    // the previous layer. The first one has 64 nodes.
    let fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);

    // Create fully connected layer 1, which has 32 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);

    // Create fully connected layer 2, which has 16 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);

    this.predictionTensor =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 3);
/*/
    // Create 4 fully connected layers. The first one has 32 nodes.
    let fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, this.inputTensor, 0, 32);

    // Create fully connected layer 1, which has 48 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 48);

    // Create fully connected layer 1, which has 16 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);

    // Create fully connected layer 2, which has 8 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 8);

    this.predictionTensor =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 4, 3);
  */
    // We will optimize using mean squared loss.
    this.costTensor =
        graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    // Create the session only after constructing the graph.
    this.session = new Session(graph, this.math);

    // Generate the data that will be used to train the model.
    this.generateTrainingData(1e5);
  }

  /**
   * Trains one batch for one iteration. Call this method multiple times to
   * progressively train. Calling this function transfers data from the GPU in
   * order to obtain the current loss on training data.
   *
   * If shouldFetchCost is true, returns the mean cost across examples in the
   * batch. Otherwise, returns -1. We should only retrieve the cost now and then
   * because doing so requires transferring data from the GPU.
   */
  train1Batch(shouldFetchCost: boolean): number {
    // Every 42 steps, lower the learning rate by 15%.
    const learningRate =
        this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));
    this.optimizer.setLearningRate(learningRate);

    // Train 1 batch.
    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
          shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

      if (!shouldFetchCost) {
        // We only train. We do not compute the cost.
        return;
      }

      // Compute the cost (by calling get), which requires transferring data
      // from the GPU.
      costValue = cost.get();
    });
    return costValue;
  }

  normalizeColor(rgbColor: number[]): number[] {
    return rgbColor.map(v => v / 255);
  }

  denormalizeColor(normalizedRgbColor: number[]): number[] {
    return normalizedRgbColor.map(v => v * 255);
  }

  predict(rgbColor: number[]): number[] {
    let complementColor: number[] = [];
    this.math.scope((keep, track) => {
      const mapping = [{
        tensor: this.inputTensor,
        data: Array1D.new(this.normalizeColor(rgbColor)),
      }];
      const evalOutput = this.session.eval(this.predictionTensor, mapping);
      const values = evalOutput.getValues();
      const colors = this.denormalizeColor(Array.prototype.slice.call(values));

      // Make sure the values are within range.
      complementColor =
          colors.map(v => Math.round(Math.max(Math.min(v, 255), 0)));
    });
    return complementColor;
  }

  private createFullyConnectedLayer(
      graph: Graph, inputLayer: Tensor, layerIndex: number,
      sizeOfThisLayer: number) {
    // Uses leakyRelu to avoid dying ReLU
    // Just a small tilt (0.001) should be enough
    return graph.layers.dense(
        `fully_connected_${layerIndex}`, inputLayer, sizeOfThisLayer,
        (x) => graph.leakyRelu(x, 0.001));
  }

  /**
   * Generates data used to train. Creates a feed entry that will later be used
   * to pass data into the model. Generates `exampleCount` data points.
   */
  private generateTrainingData(exampleCount: number) {
    this.math.scope(() => {
      const rawInputs = new Array(exampleCount);
      for (let i = 0; i < exampleCount; i++) {
        rawInputs[i] = [
          this.generateRandomChannelValue(), this.generateRandomChannelValue(),
          this.generateRandomChannelValue()
        ];
      }

      // Store the data within Array1Ds so that learnjs can use it.
      const inputArray: Array1D[] =
          rawInputs.map(c => Array1D.new(this.normalizeColor(c)));
      const targetArray: Array1D[] = rawInputs.map(
          c => Array1D.new(
              this.normalizeColor(this.computeComplementaryColor(c))));

      // This provider will shuffle the training data (and will do so in a way
      // that does not separate the input-target relationship).
      const shuffledInputProviderBuilder =
          new InCPUMemoryShuffledInputProviderBuilder(
              [inputArray, targetArray]);
      const [inputProvider, targetProvider] =
          shuffledInputProviderBuilder.getInputProviders();

      // Maps tensors to InputProviders.
      this.feedEntries = [
        {tensor: this.inputTensor, data: inputProvider},
        {tensor: this.targetTensor, data: targetProvider}
      ];
    });
  }

  private generateRandomChannelValue() {
    return Math.floor(Math.random() * 256);
  }

  /**
   * This implementation of computing the complementary color came from an
   * answer by Edd https://stackoverflow.com/a/37657940
   */
  computeComplementaryColor(rgbColor: number[]): number[] {
    let r = rgbColor[0];
    let g = rgbColor[1];
    let b = rgbColor[2];

    // Convert RGB to HSL
    // Adapted from answer by 0x000f http://stackoverflow.com/a/34946092/4939630
    r /= 255.0;
    g /= 255.0;
    b /= 255.0;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = (max + min) / 2.0;
    let s = h;
    const l = h;

    if (max === min) {
      h = s = 0;  // achromatic
    } else {
      const d = max - min;
      s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));

      if (max === r && g >= b) {
        h = 1.0472 * (g - b) / d;
      } else if (max === r && g < b) {
        h = 1.0472 * (g - b) / d + 6.2832;
      } else if (max === g) {
        h = 1.0472 * (b - r) / d + 2.0944;
      } else if (max === b) {
        h = 1.0472 * (r - g) / d + 4.1888;
      }
    }

    h = h / 6.2832 * 360.0 + 0;

    // Shift hue to opposite side of wheel and convert to [0-1] value
    h += 180;
    if (h > 360) {
      h -= 360;
    }
    h /= 360;

    // Convert h s and l values into r g and b values
    // Adapted from answer by Mohsen http://stackoverflow.com/a/9493060/4939630
    if (s === 0) {
      r = g = b = l;  // achromatic
    } else {
      const hue2rgb = (p: number, q: number, t: number) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };

      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;

      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }

    return [r, g, b].map(v => Math.round(v * 255));
  }
}

const complementaryColorModel = new ComplementaryColorModel();

// Create the graph of the model.
complementaryColorModel.setupSession();

// On every frame, we train and then maybe update the UI.
let step = 0;
let cost = +Infinity;

function trainAndMaybeRender() {
  // Stops at a certain setpLimit or costTarget, whatever happens first
  const stepLimit = 4242;
  const costTarget = 1.2e-4;
  if (step >= stepLimit) {
    finishTrainingAndRendering(`Reached step limit (${stepLimit})`);
    // Stop training.
    return;
  }
  if (cost <= costTarget) {
    finishTrainingAndRendering(`Reached cost target (${costTarget})\nCost: ${cost} Step:${step}`);
    // Stop training
    return;
  }

  function finishTrainingAndRendering(message:String):void {
    console.log(message);
    console.log(totalTime());
  }

  // Schedule the next batch to be trained.
  requestAnimationFrame(trainAndMaybeRender);

  // We only fetch the cost every 5 steps because doing so requires a transfer
  // of data from the GPU.
  const localStepsToRun = 5;
  for (let i = 0; i < localStepsToRun; i++) {
    cost = complementaryColorModel.train1Batch(i === localStepsToRun - 1);
    step++;
  }

  // Print data to console so the user can inspect.
  console.log('step', step, 'cost', cost);

  // Visualize the predicted complement.
  const colorRows = document.querySelectorAll('tr[data-original-color]');
  for (let i = 0; i < colorRows.length; i++) {
    const rowElement = colorRows[i];
    const tds = rowElement.querySelectorAll('td');
    const originalColor =
        (rowElement.getAttribute('data-original-color') as string)
            .split(',')
            .map(v => parseInt(v, 10));

    // Visualize the predicted color.
    const predictedColor = complementaryColorModel.predict(originalColor);
    populateContainerWithColor(
        tds[2], predictedColor[0], predictedColor[1], predictedColor[2]);
  }
  // Updates outer ring of predicted colors
  d3.selectAll(".predicted")
      .style("fill",
        function(d:any){
          const originalColor:number[] = [parseInt(d.data.color.slice(1,3), 16), parseInt(d.data.color.slice(3,5), 16), parseInt(d.data.color.slice(5,7), 16)];
          const predicted = complementaryColorModel.predict(originalColor);
          d.data.cost = colorCost(originalColor, predicted);
          return sharpRGBColor(predicted);
         });

  // Deletes all temporary text
  d3.select("svg").selectAll(".tempText").remove();
  // And create new labels
  d3.select("svg")
    .append("text")
      .attr("x", -120)
      .attr("y", -30)
      .attr("class", "tempText")
      .attr("font-size", "36px")
      .style("fill", "black")
      // Step Count
      .text("Step "+step);
  d3.select("svg")
    .append("text")
      .attr("x", -120)
      .attr("y",  20)
      .attr("class", "tempText")
      .attr("font-size", "36px")
      .style("fill", "black")
      // Current cost
      .text("Cost "+cost.toLocaleString("en", {maximumFractionDigits: 8}));
    d3.select("svg")
    .append("text")
      .attr("x", -120)
      .attr("y",  70)
      .attr("class", "tempText")
      .attr("font-size", "36px")
      .style("fill", "black")
      // Current cost
      .text("Time "+totalTime().toLocaleString("en", {maximumFractionDigits: 8}));
  // Updates color labels
  d3.selectAll(".predicted")
      .each(function(d:any, i:any){
        d3.select("svg")
            .append("text")
            .attr("class", "tempText txPr")
            .append("textPath")
            .attr("xlink:href", "#pr"+i)
            .append("tspan")
            .attr("dy", -10)
            .attr("dx", 95)
            .attr("text-anchor", "middle")
            .text(function(d:any){return d3.select("#pr"+i).style("fill").slice(4,-1);});
        return false;
      });

  // Estimates the cost for a single output (a label, b predicted)
  function colorCost(a:number[], b:number[]):number {
    // Normalize color channels before estimating single color loss
    return Math.pow((Math.pow((a[0]-b[0])/255, 2)+Math.pow((a[1]-b[1])/255, 2)+Math.pow((a[2]-b[2])/255, 2)),0.5)/3;
  }
}

function sharpRGBColor(color:number[]):String {
  return "#" + ("0" + color[0].toString(16)).slice(-2)
             + ("0" + color[1].toString(16)).slice(-2)
             + ("0" + color[2].toString(16)).slice(-2);
}

function populateContainerWithColor(
    container: HTMLElement, r: number, g: number, b: number) {
  const originalColorString = 'rgb(' + [r, g, b].join(',') + ')';
  container.textContent = originalColorString;

  const colorBox = document.createElement('div');
  colorBox.classList.add('color-box');
  colorBox.style.background = originalColorString;
  container.appendChild(colorBox);
}

function initializeUi() {
  var testColors:any[] = [];
  const colorRows = document.querySelectorAll('tr[data-original-color]');
  for (let i = 0; i < colorRows.length; i++) {
    const rowElement = colorRows[i];
    const tds = rowElement.querySelectorAll('td');
    const originalColor =
        (rowElement.getAttribute('data-original-color') as string)
            .split(',')
            .map(v => parseInt(v, 10));

    // Visualize the original color.
    populateContainerWithColor(
        tds[0], originalColor[0], originalColor[1], originalColor[2]);

    // Visualize the complementary color.
    const complement =
        complementaryColorModel.computeComplementaryColor(originalColor);
    populateContainerWithColor(
        tds[1], complement[0], complement[1], complement[2]);
    testColors.push({color:sharpRGBColor(originalColor), value: 42});
  }

  var svg = d3.select("svg");
  var arc = [d3.arc().innerRadius(250).outerRadius(320),
             d3.arc().innerRadius(350).outerRadius(420),
             d3.arc().innerRadius(450).outerRadius(520)];
  var pie = d3.pie().value(function(d:any){return d.value;}).sort(null);
  // Creates the inner color donut
  const widthHeight = Math.min(window.innerHeight, window.innerWidth);
  // Sets SVG parameters and draws the three color donuts
  svg.attr("width", widthHeight).attr("height", widthHeight);
  svg.selectAll("original")
        .data(pie(testColors))
        .enter()
        .append("path")
        .attr("d", arc[0])
        .style("fill", function(d:any){return d.data.color;})
        .attr("id", function(d:any, i:any){return "or"+i;})
        .attr("class", "original");
  // Creates the middle complementary color donut
  svg.selectAll("complement")
        .data(pie(testColors))
        .enter()
        .append("path")
        .attr("d", arc[1])
        .style("fill", function(d:any){return actualComplement(d.data.color);})
        .attr("id", function(d:any, i:any){return "co"+i;})
        .attr("class", "complement");
  // Creates the outer predicted color donut
  svg.selectAll("predicted")
        .data(pie(testColors))
        .enter()
        .append("path")
        .attr("d", arc[2])
        .style("fill", sharpRGBColor([255,255,255]))
        .attr("id", function(d:any, i:any){return "pr"+i;})
        .attr("class", "predicted");
  // Creates the labels for the original colors
  d3.selectAll(".original")
    .each(function(d:any, i:any){
      svg.append("text")
          .attr("class", "txOr")
          .append("textPath")
          .attr("xlink:href", "#or"+i)
          .append("tspan")
          .attr("dy", -10)
          .attr("dx", 58)
          .attr("text-anchor", "middle")
          .text(function(d:any){return d3.select("#or"+i).style("fill").slice(4,-1);});
      return false;
    });
  // Creates the labels for the complementary colors
  d3.selectAll(".complement")
    .each(function(d:any, i:any){
      svg.append("text")
          .attr("class", "txCo")
          .append("textPath")
          .attr("xlink:href", "#co"+i)
          .append("tspan")
          .attr("dy", -10)
          .attr("dx", 78)
          .attr("text-anchor", "middle")
          .text(function(d:any){return d3.select("#co"+i).style("fill").slice(4,-1);});
      return false;
    });

  // Add annotations
  const annotations = [{
    note: { label: "Original colors (RGB)"},
    x: 272, y: -212,
    dy: -235, dx: 105
  },{
    note: { label: "Computed complementary colors (RGB)"},
    x: 425, y: -130,
    dy: -160, dx: 55
  }];

  const makeAnnotations = d3.annotation().annotations(annotations);
  d3.select("svg").append("g").attr("class", "annotation-group").call(makeAnnotations);

  setTimeout(function(){
    const annotationG = [{
      note: { label: "Generated colors (RGB)"},
      x: 438, y: -328,
      dy: -80, dx: 65
    }], makeAnnotationG = d3.annotation().annotations(annotationG);

    d3.select("svg").append("g").attr("class", "annotation-group").call(makeAnnotationG);
  }, 10);
  // setTimeout(function(){d3.selectAll(".annotations").style("opacity", 0);}, 300000);

  // Computes the complementary color of an input string in the #rrggbb format and returns
  // the complementary color on the same notation
  function actualComplement(color:String):String {
    const originalColor:number[] = [parseInt(color.slice(1,3), 16), parseInt(color.slice(3,5), 16), parseInt(color.slice(5,7), 16)];
    const complement = complementaryColorModel.computeComplementaryColor(originalColor);
    return sharpRGBColor(complement);
  }
}

var startTrainingTime = new Date();
// Total time in minutes
function totalTime() {
  var now = new Date;
  return (now.getTime() - startTrainingTime.getTime())/60000;
}
// Kick off training.
initializeUi();
// Calling requestAnimationFrame through setTimeout
// gives room for the browser screen initialization
setTimeout(function():any{requestAnimationFrame(trainAndMaybeRender);});


// Create tensors for meanSquared cost computation in trainAndMaybeRender!!