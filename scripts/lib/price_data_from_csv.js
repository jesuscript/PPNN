const csv = require('csvtojson'),
      fs = require('fs');

const inputFileName = process.argv[2],
      outputFileName = process.argv[3]

if (!(inputFileName && outputFileName)) throw new Error("input and output files must be specified")

var avgOver = 24;


var getVals = (r,v)=> {
  return r.map(r => Number(r[v]))
}

var absToRel = (vals) => vals.map((v,i) => {
  if(!i || !vals[i-1]){
    return 0
  }else{
    return  v / vals[i-1] -  1
  }
})

var scale = (vals) => {
  var min= Math.min(...vals),
      max= Math.max(...vals)

  return vals.map(v => {
   return (v - min) / (max - min) 
  })
}

var avgNext = (vals,num) =>{
  return vals.map((v,i) => {
    if(vals.length > num+i){
      return vals.slice(i+1,i+num+1).reduce((sum,v) => sum+v) / num
    }else{
      return 0
    }
  })
}


csv()
  .fromFile(inputFileName)
  .then((records)=> {
    //records = records.slice(0,50)

    var normClose = scale(absToRel(getVals(records,"Close"))),
        normVol = scale(getVals(records,"Volume To")),
        normNextAvg = avgNext(getVals(records,"Close"), 24)


    console.log(normNextAvg)

    // dataSet = dataSet.map(r => r.map((v,i) => {
    // }).slice(1)) //abs to rel

    // console.log(dataSet);

    // dataSet = dataSet.map(r => ({
    //   min: Math.min(...r),
    //   max: Math.max(...r),
    //   data: r
    // })).map(r => {
    //   return r.data.map(v => (v - r.min) / (r.max - r.min)) 
    // }) //scaling

    // console.log(dataSet);

    // var trainingData = dataSet[0].map((v,i) => {
    //   return {
    //     //target: (i>0 && v>dataSet[0][i-1]) ? [1,0] : [0,1],
    //     target: [v],
    //     inputs: Array.prototype.concat(...dataSet.map(r => r[i]))
    //   }      
    // })

    // //console.log(trainingData)
    
    // fs.writeFile(outputFileName, JSON.stringify(trainingData), err => {
    //   if(err) {
    //     throw err  
    //   }else{
    //     console.log(`Data written to ${outputFileName}`);
    //   }
    // });
  })
