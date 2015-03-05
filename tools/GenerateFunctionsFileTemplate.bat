echo off
echo "This will overwrite the existing functions.c file!"
pause
XSLTProcessor.exe XMLModelFile.xml functions.xslt functions.c
pause