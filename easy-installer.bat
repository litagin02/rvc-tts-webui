@echo off

:: Downloading models in root directory
echo Downloading models in root directory...
call curl -L -O https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
call curl -L -O https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
echo Done!

echo.

:: Make virtual environment
echo Making virtual environment...
call python -m venv venv
echo Done!

:: Activate venv (for Windows)
echo Activating virtual environment...
call venv\Scripts\activate
echo Done!

:: Install PyTorch manually if you want to use NVIDIA GPU (Windows)
:: See https://pytorch.org/get-started/locally/ for more details
echo Installing PyTorch...
call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo Done!

echo.

:: Install requirements
echo Installing requirements...
call pip install -r requirements.txt
echo Done!