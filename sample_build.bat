cd media
..\Utils\OOCTree.exe -i .\sponza_obj\sponza.obj
..\Utils\PostProcesses .\sponza_obj\sponza.obj "REMOVE_INDEX|HCCMESH|ASVO|GPU" 8 10 5
cd ..