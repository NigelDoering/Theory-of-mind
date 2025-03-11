@echo off
echo Creating TOM Structure

mkdir src
mkdir src\simulation
mkdir src\environment
mkdir src\agents
mkdir src\planning
mkdir src\utils
mkdir examples

echo. > src\__init__.py
echo. > src\simulation\__init__.py
echo. > src\environment\__init__.py
echo. > src\agents\__init__.py
echo. > src\planning\__init__.py
echo. > src\utils\__init__.py
echo. > examples\__init__.py

echo # Theory of Mind Simulation > README.md
echo Done! 