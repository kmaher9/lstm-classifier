var fs = require('fs')                           // to get directory listings, and read files - npm install fs
var brain = require('brain.js')                  // contains core functions - npm install brain.js

var businessPath = "path/to/business/dataset"    // location of 100 training files
var techPath = "path/to/tech/dataset"            // second class 100 training files
var testFile = "path/to/test/file.txt"           // file which will be used to test accuracy - belongs to tech

var inputs = []                                  // will be used to store all training data prior to training

var errorThreshold = 0.085                       // the minimum error margin needed to class the nn as trained
var iterations = 100                             // the maximum number of training iterations to meet, if the errorThreshold is not met

var net = new brain.recurrent.LSTM()             // the instance of the LSTM model I will be using

var businessFiles = fs.readdirSync(businessPath) // retrieve directory listing
var techFiles = fs.readdirSync(techPath)

for (var i = 0; i < businessFiles.length; i++) {
    var file = businessFiles[i]

    var content = fs.readFileSync(businessFiles + "\\" + file) // opens the file as a stream, reading the whole file into memory
    
    inputs.push({input: content.toString(), output: 'business' }) // the entire file content is stored in a two dimensional array
}

for (var i = 0; i < techFiles.length; i++) {
    var file = techFiles[i]

    var content = fs.readFileSync(techFiles + "\\" + file)
    
    inputs.push({input: content.toString(), output: 'tech' })
}


net.train(inputs, { 
  log: true, // prints to the screen the error rate, and current iteration
  errorThresh: 0.05, // a relatively high error margin - but I'm not shooting for 100% accuracy (and it's merely a test)
})

var testFileContent = fs.readFileSync(testFile) // read the entire sample file into memory
var output = net.run(testFileContent) // and then run it! This will predict the class for the testFile
console.log(output)

// output: 
// [ business: 0.0043, tech: 0.8552]
