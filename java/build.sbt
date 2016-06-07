name := "Sca" // change to project name
organization := "de.fuberlin.mi.biorobotics" // change to your org
version := "0.1"
scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  // spark core
  "org.apache.spark" %% "spark-core" % "1.6.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.6.1" % "provided",
  "org.apache.spark" %% "spark-hive" % "1.6.1" % "provided",

  // spark-modules
  // "org.apache.spark" %% "spark-graphx" % "1.6.0",
  // "org.apache.spark" %% "spark-mllib" % "1.6.0",
  // "org.apache.spark" %% "spark-streaming" % "1.6.0",

  // spark packages
  "com.databricks" %% "spark-csv" % "1.3.0",

  // testing
  "org.scalatest"   %% "scalatest"    % "2.2.4"   % "test",
  "org.scalacheck"  %% "scalacheck"   % "1.12.2"      % "test",

  // logging
  "org.apache.logging.log4j" % "log4j-api" % "2.4.1",
  "org.apache.logging.log4j" % "log4j-core" % "2.4.1",

  // capnproto
  "org.capnproto" % "runtime" % "0.1.1"
)

// allows us to include spark packages
resolvers += "bintray-spark-packages" at
  "https://dl.bintray.com/spark-packages/maven/"

resolvers += "Typesafe Simple Repository" at
 "http://repo.typesafe.com/typesafe/simple/maven-releases/"

mainClass in (Compile, packageBin) := Some("de.fuberlin.biorobotics.test.ExampleClass")

// Compiler settings. Use scalac -X for other options and their description.
// See Here for more info http://www.scala-lang.org/files/archive/nightly/docs/manual/html/scalac.html
scalacOptions ++= List("-feature","-deprecation", "-unchecked", "-Xlint")

// ScalaTest settings.
// Ignore tests tagged as @Slow (they should be picked only by integration test)
testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-l",
  "org.scalatest.tags.Slow", "-u","target/junit-xml-reports", "-oD", "-eS")



sourceGenerators in Compile += Def.task {
  val prefix = file("..") / "bb_binary"
  val schema = file("..") / "bb_binary" / "bb_binary_schema.capnp"
  val output_dir = (sourceManaged in Compile).value / "java"
  output_dir.mkdirs()
  println(schema)
  val capnp_command = s"capnp compile -ojava:$output_dir  --src-prefix=$prefix $schema"
  println(capnp_command)
  capnp_command !
  val pathFinder = output_dir **  "*.java"
  println(pathFinder.get)
  pathFinder.get
}.taskValue
