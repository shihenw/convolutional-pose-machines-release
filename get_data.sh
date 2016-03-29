# get FLIC
wget http://vision.grasp.upenn.edu/video/FLIC.zip
unzip FLIC.zip -d dataset/
rm -f FLIC.zip

# get LEEDS
cd dataset/; mkdir LEEDS/; cd ..
wget http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset.zip
wget http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip
unzip lsp_dataset.zip -d dataset/LEEDS/
cd dataset/LEEDS/; mkdir lspet_dataset; cd ../..
unzip lspet_dataset.zip -d dataset/LEEDS/lspet_dataset
rm -f lsp_dataset.zip
rm -f lspet_dataset.zip

# get MPI
mkdir dataset/MPI/
wget http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
wget http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz
tar -zxvf mpii_human_pose_v1.tar.gz -C dataset/MPI/
tar -zxvf mpii_human_pose_v1_u12_1.tar.gz -C dataset/MPI/
rm -f mpii_human_pose_v1.tar.gz
rm -f mpii_human_pose_v1_u12_1.tar.gz
