import importlib

# List of libraries to test for installation
libraries = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "spectral",
    "opencv-python-headless",
    "plotly",
    "torch",
    "torchvision",
    "tensorflow",
    "h5py",
]

def test_libraries():
    all_success = True
    print("Testing library installations...\n")
    for library in libraries:
        try:
            importlib.import_module(library.replace('-', '_'))
            print(f"{library}: Installed")
        except ImportError:
            print(f"{library}: Not Installed")
            all_success = False
        except Exception as e:
            print(f"{library}: Error occurred - {e}")
            all_success = False

    print("\nTest Result:")
    if all_success:
        print("All libraries are installed successfully!")
    else:
        print("Some libraries are missing or have errors.")

if __name__ == "__main__":
    test_libraries()
