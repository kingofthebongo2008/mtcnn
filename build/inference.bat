@echo off


echo ^<?xml version="1.0" encoding="utf-8"?^> > inference_app_h.msbuild
echo ^<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^> >> inference_app_h.msbuild
echo ^<ItemGroup^> >> inference_app_h.msbuild

ucdev_build_file_generator_r.exe --input ..\src\inference\private\ --mode h >> inference_app_h.msbuild
ucdev_build_file_generator_r.exe --input ..\include\ --mode h >> inference_app_h.msbuild

echo ^</ItemGroup^> >> inference_app_h.msbuild
echo ^</Project^> >> inference_app_h.msbuild

echo ^<?xml version="1.0" encoding="utf-8"?^> > inference_app_cpp.msbuild
echo ^<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^> >> inference_app_cpp.msbuild
echo ^<ItemGroup^> >> inference_app_cpp.msbuild

ucdev_build_file_generator_r.exe --input ..\src\inference\private\ --mode cpp >> inference_app_cpp.msbuild

echo ^</ItemGroup^> >> inference_app_cpp.msbuild
echo ^</Project^> >> inference_app_cpp.msbuild





