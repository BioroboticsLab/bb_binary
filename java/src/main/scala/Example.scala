package de.fuberlin.biorobotics.test

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import scala.io.Source
import scala.collection.JavaConverters._

import org.capnproto.{MessageReader, PackedInputStream, SerializePacked, Serialize}
import de.fuberlin.biorobotics.BeesBook.{FrameContainer, Frame, DetectionDP}

case class Detection(
    tagIdx       : Int,    // unique sequential id of the tag
    candidateIdx : Int,    // sequential id of the candidate per tag
    gridIdx      : Int,    // sequential id of the grid/decoding per candidate
    xpos         : Int,    // x coordinate of the grid center
    ypos         : Int,    // y coordinate of the grid center
    xRotation    : Float,  // rotation of the grid in x plane
    yRotation    : Float,  // rotation of the grid in y plane
    zRotation    : Float,  // rotation of the grid in z plane
    lScore       : Float,  // roi score
    eScore       : Float,  // ellipse score
    gScore       : Float,  // grid score
    id           : Int     // decoded id
)

object ExampleClass extends Serializable {
  def detectionFromRow(row: Seq[String]): Detection = {
    // Converts a csv row into a Detection
    Detection(
      row(0).toInt,
      row(1).toInt,
      row(2).toInt,
      row(3).toInt,
      row(4).toInt,
      row(5).toFloat,
      row(6).toFloat,
      row(7).toFloat,
      row(8).toFloat,
      row(9).toFloat,
      row(10).toFloat,
      Integer.parseInt(row(11), 2))
  }
  def load_capnp(fname: String): Seq[Detection] = {
    val message = Serialize.read(new java.io.FileInputStream(fname).getChannel)
    val fc = message.getRoot(FrameContainer.factory)
    val frames = fc.getFrames
    val frame: Frame.Reader = frames.get(0)
    val detections = frame.getDetectionsUnion.getDetectionsDP
    for(d: DetectionDP.Reader <- detections.asScala) {
      println(d.getIdx)
      println(d.getXRotation)
      println(d.getYRotation)
      println(d.getDecodedId)
    }
    return Seq()
  }

  def load_csv(fname: String): Seq[Detection] = {
    // instead of loading csv files we could also deserialize flatbuffers
    val src = Source.fromFile(fname).getLines()
    return src.map(_.split(",").toSeq).map(detectionFromRow).toSeq
  }

  def main(args:Array[String]) = {
    val name = "Application"
    val conf = new SparkConf().setAppName(name)
    val sc = SparkContext.getOrCreate(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    import sqlContext.implicits._

    // instead of this ugly absolute path we will use some generic way to
    // comunicate with the nas.
    val basename = "/home/leon/uni/bioroboticslab/bb_binary/tests/data/"
    // the actual csv files are in the csv_files.txt
    val csv_files = sc.textFile(basename + "csv_files.txt")
    val abs_fnames = csv_files.map(basename + _)
    // now load all detection in the csv files
    val detections = abs_fnames.flatMap(load_csv)
    val detections_df = detections.toDF()
    detections_df.registerTempTable("detections")
    val detections_left_top = sqlContext.sql(
      "SELECT id, xpos, ypos FROM detections WHERE xpos <= 1250 AND ypos <= 1250")
    println(detections_left_top.count())
    detections_left_top.map(t => t.mkString(",")).collect().foreach(println)
  }
}
