name := """parametric-image-generator"""
version       := "1.0"

scalaVersion  := "2.12.17"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers += Resolver.jcenterRepo

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")

libraryDependencies += "ch.unibas.cs.gravis" %% "scalismo-faces" % "0.90.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.14" % "test"

libraryDependencies += "org.rogach" %% "scallop" % "4.1.0"

mainClass in assembly := Some("faces.apps.ListApplications")

assemblyJarName in assembly := "generator.jar"
