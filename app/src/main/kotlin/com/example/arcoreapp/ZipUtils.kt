package com.example.arcoreapp

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

object ZipUtils {
    fun zipDirectory(directory: File, zipFile: File) {
        ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile))).use { zos ->
            directory.walkTopDown().filter { it.isFile }.forEach { file ->
                val entryName = file.relativeTo(directory).path
                zos.putNextEntry(ZipEntry(entryName))
                BufferedInputStream(FileInputStream(file)).use { bis ->
                    bis.copyTo(zos)
                }
                zos.closeEntry()
            }
        }
    }
}
