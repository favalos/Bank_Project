package com.favalos.bank

import java.io.FileWriter

import com.github.tototoshi.csv.CSVWriter
import org.apache.mxnet.Symbol.Variable

import org.apache.mxnet.optimizer.SGD
import org.apache.mxnet.{Accuracy, Context, FeedForward, IO, Symbol, Xavier}


object BankClassification {


  def loadData(file: String) = {

    val dataRaw = io.Source.fromInputStream(BankClassification.getClass.getResourceAsStream(file))
    val data = dataRaw.getLines().map(_.split(",")).toArray

    data.tail
  }

  def dataToNumeric(arr: Array[Array[String]]): Array[Array[String]] = {

    transformFactor(1, arr)
    transformFactor(2, arr)
    transformFactor(3, arr)
    transformFactor(4, arr)
    transformFactor(5, arr)
    transformFactor(6, arr)
    transformFactor(7, arr)
    transformFactor(8, arr)
    transformFactor(9, arr)
    transformFactor(14, arr)

    arr
  }

  def transformFactor(col: Int, arrData: Array[Array[String]]): Array[Array[String]] = {

    val colData = arrData.map(_(col))
    val factorMap = colData.distinct.zipWithIndex.toMap

    arrData.map { arr =>
      arr(col) = factorMap.getOrElse(arr(col), -1).toString
      arr
    }

  }

  def prepareData() = {

    val data = loadData("banking.csv")

    val numericData = dataToNumeric(data)
    val splittedData = numericData.map(_.slice(0, 20)).splitAt(40000)
    val splittedLabel = numericData.map(_.slice(20, 21)).splitAt(40000)

    fileWriterCSV("trainData.csv", splittedData._1 )
    fileWriterCSV("trainLabel.csv", splittedLabel._1)
    fileWriterCSV("testData.csv", splittedData._2)
    fileWriterCSV("testLabel.csv", splittedLabel._2)

  }

  def fileWriterCSV(fileName: String, data: Array[Array[String]]) = {

    val f = new FileWriter(fileName)

    val writer = CSVWriter.open(f)

    writer.writeAll(data.map(_.toSeq).toSeq)

    f.flush()
    f.close()
  }

  def createTemplate(data: Symbol, label: Symbol): Symbol = {

    val fcl2 = Symbol.FullyConnected("fcl2")()(Map("data" -> data, "num_hidden" -> 128))
    val act2 = Symbol.Activation("act2")()(Map("data" -> fcl2, "act_type" -> "relu"))
    val fcl3 = Symbol.FullyConnected("fcl3")()(Map("data" -> fcl2, "num_hidden" -> 2))
    val act3 = Symbol.SoftmaxOutput("act3")()(Map("data" -> fcl3, "label" -> label))

    act3
  }

  def main(args: Array[String]): Unit = {

    prepareData()

    val trainIter = IO.CSVIter(Map("data_csv" -> "trainData.csv", "label_csv" -> "trainLabel.csv", "data_shape" -> "(20)", "batch_size" -> "500"))
    val testIter = IO.CSVIter(Map("data_csv" -> "testData.csv", "label_csv" -> "testLabel.csv", "data_shape" -> "(20)", "batch_size" -> "500"))

    val data = Variable("data")
    val label = Variable("label")

    val template = createTemplate(data, label)

    val metric = new Accuracy()

    val model = FeedForward.newBuilder(template)
                      .setContext(Array(Context.cpu()))
                      .setInitializer(new Xavier())
                      .setOptimizer(new SGD(learningRate = 0.001f, momentum = 0.8f, wd = 0.0001f))
                      .setNumEpoch(5)
                      .setTrainData(trainIter)
                      .setEvalData(testIter)
                      .setEvalMetric(metric)
                      .build()

    metric.get._2.foreach(println)
  }

}
