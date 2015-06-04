All steps can be easily done by using "EZBuilder" which is a launcher for following components with intuitive parameter settings.

(1) Build "OOC" file
To convert "PLY" or "OBJ" to "OOC" file format, use "OpenIRT\Utils\OOCTree.exe" as following:

 - OOCTree.exe [-ifmx] input_file [mtl file]
  * -i : In-core build. Without this option, the building is performed in out-of-core mode which is much efficient for massive models. Note: The out-of-core mode does not support multiple input files.
  * -f : Input_file is a file list. Multiple input files can be support by the file list.
  * -m : Apply 4x4 transform matrix to each model (requires file list).
  * -x : Assign a material index to each model.
 - (Example1) OOCTree.exe -i sponza.obj
 - (Example2) OOCTree.exe -ifm sponza sponza.mtl
  * [In "sponza" file]
  * .\sponza0.ply 2 0 0 0 0 2 0 0 0 0 2 0 0 0 0 1
  * .\sponza1.ply 2 0 0 0 0 2 0 0 0 0 2 0 0 0 0 1

Since "PLY" file format does not have standard material format, we use "MTL" file format which usually be used with "OBJ" file format together.
To apply a material for a "PLY" file, add a comment, "comment used material = material_name". The "material_name" is a name defined in the input "MTL" file.
Please note that a "PLY" file can have only single material by above process.
The output (which includes several files) will be placed in a directory named "input_file.ooc".
The original "OOC" file has unused index list (which is used to support multiple triangles for a tree node) and the representation layouts are not optimized to GPU. 
For better performance, OpenIRT uses more efficient representation by removing the index and changing layout. This is done by using following:
 - "PostProcesses.exe input_file "REMOVE_INDEX | GPU"".

(2) Build "HCCMesh", "ASVO", and other post processes
Since T-ReX renderer requires "HCCMesh" and "ASVO", we need to perform further processing for the T-ReX renderer. These are done as following:
 - "PostProcess.exe input_file "REMOVE_INDEX | HCCMESH | ASVO | GPU" [ASVO_option_1] [ASVO_option_2] [ASVO_option_3]
 - 2^[ASVO_option_1] = r_u (See T-ReX paper)
 - 2^[ASVO_option_2] = r_u*r_l
 - [ASVO_option_3] = depth of overlapped upper ASVO
 - (Example) PostProcesses.exe sponza.obj "REMOVE_INDEX|HCCMESH|ASVO|GPU" 8 10 5


[Contact information]
Tae-Joon Kim
PhD, Department of Computer Science, KAIST
Republic of Korea
tjkim.kaist@gmail.com; taejoonkim@etri.re.kr 

Myungbae Son
MS student, Department of Computer Science, KAIST
Republic of Korea
nedsociety@gmail.com 

