const csv = require('csvtojson'),
      fs = require('fs');

const inputFileName = process.argv[2],
      outputFileName = process.argv[3]

if (!(inputFileName && outputFileName)) throw new Error("input and output files must be specified")
  


csv()
  .fromFile(inputFileName)
  .then((records)=> {
    //records = records.slice(0,10)
    
    var dataSet = [
      "Close",
      "Volume To"
    ].map(f => records.map(r => Number(r[f]))) // subset
        .map(r => r.map((v,i) => {
          if(!i || !r[i-1]){
            return 1
          }else{
            return  v / r[i-1] -  1
          }
        }).slice(1)) //abs to rel
        .map(r => ({
          min: Math.min(...r),
          max: Math.max(...r),
          data: r
        }))
        .map(r => {
          return r.data.map(v => (v - r.min) / (r.max - r.min)) 
        }) //scaling

    //console.log(dataSet);

    var trainingData = dataSet[0].map((v,i) => ({
      target: (i>0 && v>dataSet[0][i-1]) ? [1,0] : [0,1],
      inputs: Array.prototype.concat(...dataSet.map(r => r[i]))
    }))

    //console.log(trainingData)
    
    fs.writeFile(outputFileName, JSON.stringify(trainingData), err => {
      if(err) {
       throw err  
      }else{
        console.log(`Data written to ${outputFileName}`);
      }
    });
  })
