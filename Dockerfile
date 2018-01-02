FROM pytorch/pytorch
LABEL maintainer="Mustafa Taha Kocyigit <taha.kocyigit@gmail.com>"

RUN apt-get update && apt-get install -y \ 
    libglib2.0-0 \ 
    libsm6 \
    libxrender1 \
    libxext6

RUN apt-get update --fix-missing && apt-get install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN pip install opencv-python jupyter h5py progress matplotlib

EXPOSE 8888

WORKDIR /pytorch-pose-hg-3d

# CMD [ "python src/demo.py -demo images/h36m_1214.png -loadModel models/hgreg-3d.pth" ]