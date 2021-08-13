cd ~
yes | sudo apt update
yes | sudo apt-get upgrade
yes | sudo apt install make gcc unzip
yes | sudo apt-get update
yes | sudo apt-get install libxt6 libxrender1 libgl1-mesa-glx libqt5widgets5 
wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.2-linux-x86_64.tar.gz | tar -xzv
export PATH=~/anaconda3/bin:~/julia-1.6.2/bin:$PATH
sudo echo 'export PATH=~/julia-1.6.2/bin:$PATH' >> .bashrc
julia --threads auto -e "using Pkg; Pkg.add(\"IJulia\")"

# Go back to old directory.
cd -
pip install -r requirements.txt
cd lectures
julia --project --threads auto -e "using Pkg; Pkg.instantiate()"