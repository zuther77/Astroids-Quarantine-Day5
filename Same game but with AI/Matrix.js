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
