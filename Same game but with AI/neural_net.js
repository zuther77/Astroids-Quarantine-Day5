"use strict";


// neural network contains 1 input layer, 1 hidden layer, 1 output layer
// each layer has nodes connected by neurons.  
// each neuron has random weights. input-hidden weights1 , hidden-output weights2
//input layer contains bias node
class network{
    constructor(inputNodes, hiddenNodes, outputNodes){
        this._inputs = [];
        this._hidden = [];
        this._inputNodes  = inputNodes;
        this._hiddenNodes = hiddenNodes;
        this._outputNodes = outputNodes;
        this._bias1 = new Matrix(1, this._hiddenNodes);
        this._bias2 = new Matrix(1, this._outputNodes);
        this._weights1 = new Matrix(this._inputNodes, this._hiddenNodes);
        this._weights2 = new Matrix(this._hiddenNodes, this._outputNodes);
        this._count = 0 ;
        //random weights generation
        this._weights1.randWeights();
        this._weights2.randWeights();
        this._bias1.randWeights();
        this._bias2.randWeights();
    }
    //getters and setters
    get inputs() {
        return this._inputs;
    }
    set inputs(inputs) {
        this._inputs = inputs;
    }
    get hidden() {
        return this._hidden;
    }
    set hidden(hidden) {
        this._hidden = hidden;
    }
    get weights1(){
        return this._weights1;
    }
    set weights1(weights) {
        this._weights1 = weights;
    }
    get weights2() {
        return this._weights2;
    }
    set weights2(weights) {
        this._weights2 = weights;
    }
    get bias1() {
        return this._bias1;
    }
    set bias1(bias) {
        this._bias1 = bias;
    }
    get bias2() {
        return this._bias2;
    }
    set bias2(bias) {
        this._bias2 = bias;
    }
    get count() {
        return this._count;
    }
    set count(count) {
        this._count = count;
    }


    feedForward(inputArr){
        // get input nodes 
        this.inputs = Matrix.OneDimArray(inputArr);

        // y = x1w1 + x1w2 + x3w3 + ... + xnwn 
        // get hidden layer and add bias1 to each node
        this.hidden = Matrix.dotProduct(this.inputs, this.weights1);
        this.hidden = Matrix.add(this.hidden , this.bias1);
        //sigmoid
        this.hidden = Matrix.map(this.hidden, x => sigmoid(x));

        // get output layer and add bias2 to each node
        let outputs = Matrix.dotProduct(this.hidden, this.weights2);
        outputs = Matrix.add(outputs,this.bias2);
        // sigmoid
        outputs = Matrix.map(outputs, x => sigmoid(x));

        return outputs;
    }

    // to train is to find error between the target output and output from network
    // minimise this error to get true automation
    
    train(inputArr , targetOutputArr){
        this.count++;
        // feed the input data through the network
        let outputs = this.feedForward(inputArr);

        // calculate the output errors (target Output  - output from network)
        let  target = Matrix.OneDimArray(targetOutputArr);
        let error = Matrix.sub(target , outputs);
        if(this.count%5000==0){
            console.log("error "+ error.data[0][0]);
        }

        // calculate delta i.e error * derivative of output
        let outDerivatives = Matrix.map( outputs , x => sigmoid(x,true));
        let outDelta = Matrix.mul( error , outDerivatives);

        // calculate hidden layer errors ( deltas dot with transpose of weights)
        let weights2T = Matrix.trans(this.weights2);
        let hidErr = Matrix.dotProduct(outDelta, weights2T);

        // cal culate hidden layer delta ( hidErr * derivative of hidden)
        let hiddenDerivatives = Matrix.map( this.hidden , x => sigmoid(x,true));
        let hidDelta = Matrix.mul( hidErr , hiddenDerivatives);

        // update the weights2 (weights2 +  transpose of hidden layers dot out layer delta)
        let hiddenT = Matrix.trans(this.hidden);
        this.weights2 = Matrix.add(this.weights2 , Matrix.dotProduct(hiddenT, outDelta));

        // update the weights1 (weights1 +  transpose of hidden layers dot out layer delta)
        let inputT = Matrix.trans(this.inputs);
        this.weights1 = Matrix.add(this.weights1 , Matrix.dotProduct(inputT, hidDelta));
        
        // update bias
        this.bias2 = Matrix.add(this.bias2, outDelta);
        this.bias1 = Matrix.add(this.bias1, hidDelta);
    }


}

function sigmoid(x , derivative = false){
    if(derivative){
        return x * (1 - x);         // here x = sigmoid(x) since we apply sigmoid in feedForward
    }
    return 1 / ( 1 + Math.exp(-x));
}


class Matrix {
    constructor(rows, cols, data = []) {
        this._rows = rows;
        this._cols = cols;
        this._data = data;
        if (data == null || data.length == 0) {
            this._data = [];
            for (let i = 0; i < this._rows; i++) {
                this._data[i] = [];
                for (let j = 0; j < this._cols; j++) {
                    this._data[i][j] = 0;
                }
            }
        } else {
            if (data.length != rows || data[0].length != cols) {
                throw new Error("Incorrect data dimensions!");
            }
        }
    }

    get rows() {
        return this._rows;
    }
    get cols() {
        return this._cols;
    }
    get data() {
        return this._data;
    }

    // matrix addition
    static add(m1,m2){
        Matrix.checkDimension(m1,m2);
        let res = new Matrix(m1.rows, m1.cols);
        for (let i = 0; i < res.rows; i++) {
            for (let j = 0; j < res.cols; j++) {
                 res.data[i][j] = m1.data[i][j] + m2.data[i][j];
            }
        }
        return res;
    }

    // matrix sub
    static sub(m1,m2){
        Matrix.checkDimension(m1,m2);
        let res = new Matrix(m1.rows, m1.cols);
        for (let i = 0; i < res.rows; i++) {
            for (let j = 0; j < res.cols; j++) {
                 res.data[i][j] = m1.data[i][j] - m2.data[i][j];
            }
        }
        return res;
    }

    // matrix multiply
    static mul(m1,m2){
        Matrix.checkDimension(m1,m2);
        let res = new Matrix(m1.rows, m1.cols);
        for (let i = 0; i < res.rows; i++) {
            for (let j = 0; j < res.cols; j++) {
                 res.data[i][j] = m1.data[i][j] * m2.data[i][j];
            }
        }
        return res;
    }

    // matrix transpose 
    static trans(x) {
        let res = new Matrix(x.cols, x.rows);
        for (let i = 0; i < x.rows; i++) {
            for (let j = 0; j < x.cols; j++) {
                res.data[j][i] = x.data[i][j];
            }
        }
        return res;
    }

    //check if matrix m1 and m2 can be added or subtracted or not 
    static checkDimension(m1, m2) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
            throw new Error("Different dimensions! Cannot add");
        }
    }

    //dot product
    static dotProduct(m1,m2){
        if(m1.cols != m2.rows) {
            throw new Error("Dot product cannot be performed");
        }
        let res = new Matrix(m1.rows , m2.cols);
        for (let i = 0; i < res.rows; i++) {
            for (let j = 0; j < res.cols; j++) {
                let sum = 0 ;
                for (let k = 0; k < m1.cols; k++) {
                    sum += m1.data[i][k] * m2.data[k][j];
                }  
                res.data[i][j] = sum;
            }
        }
        return res;
    }




    // array to 1D matrix
    static OneDimArray(arr){
        return new Matrix( 1 , arr.length , [arr]);
    }

    // apply sigmoid function to each cell
    static map(m , actiFunction){
        let m0 = new Matrix(m.rows, m.cols);
        for (let i = 0; i < m0.rows; i++) {
            for (let j = 0; j < m0.cols; j++) {
                m0.data[i][j] = actiFunction(m.data[i][j]);
            }
        }
        return m0
    }

    // create random weights for network
    randWeights() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }


}
