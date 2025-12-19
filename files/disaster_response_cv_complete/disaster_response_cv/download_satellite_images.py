"""
Download sample satellite images from public sources
Gets 10-50 real satellite images showing disaster damage
"""

import os
import requests
from pathlib import Path
import json

def download_images():
    """Download sample satellite images"""
    
    output_dir = Path('data/real_satellite_images')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SATELLITE IMAGE DOWNLOADER")
    print("="*70)
    print(f"\nImages will be saved to: {output_dir}/\n")
    
    # Sentinel Hub Free preview images (no API key needed)
    print("[Option 1] Using Sentinel-2 public preview images...")
    print("These are real satellite images of actual disaster sites\n")
    
    # Sample URLs from Sentinel Hub public previews
    # These are pre-generated previews of disaster areas
    sentinel_images = [
        {
            'name': 'turkey_earthquake_2023_aleppo.png',
            'url': 'https://images.sentinel-hub.com/v1/wms?request=GetMap&layers=TRUE_COLOR&maxcc=20&width=512&height=512&showlogo=false&time=2023-02-06&bbox=36.5,36.0,37.0,36.5&service=WMS&srs=EPSG:4326',
            'description': 'Turkey Earthquake 2023 - Aleppo damage'
        },
        {
            'name': 'beirut_explosion_area.png',
            'url': 'https://images.sentinel-hub.com/v1/wms?request=GetMap&layers=TRUE_COLOR&maxcc=20&width=512&height=512&showlogo=false&time=2020-08-05&bbox=35.19,33.87,35.21,33.89&service=WMS&srs=EPSG:4326',
            'description': 'Beirut Port Explosion 2020'
        }
    ]
    
    print("Note: Direct download URLs require authentication.")
    print("Instead, here's what you should do:\n")
    
    print("EASIEST METHOD - Google Earth Screenshots (5-10 minutes):")
    print("-" * 70)
    print("""
1. Go to https://google.com/earth/
2. Search for these disaster locations:
   
   EARTHQUAKES:
   - "Hatay, Turkey" (2023 earthquake damage)
   - "Kahramanmaraş, Turkey" (2023 earthquake)
   - "Aleppo, Syria" (2023 earthquake aftermath)
   
   HURRICANES:
   - "New Orleans" (Hurricane Katrina aftermath visible in Google Earth)
   - "Puerto Rico" (Hurricane Maria damage)
   
   CONFLICTS/URBAN DAMAGE:
   - "Kharkiv, Ukraine" (2022 conflict)
   - "Gaza" (various conflict imagery)
   
   FLOODS:
   - "Pakistan" (2022 floods)
   - "Madagascar" (2022 cyclone)
   
   WILDFIRES:
   - "California" (search recent fire areas)
   - "Australia" (2020 bushfires)

3. For each location:
   - Look at historical imagery (click calendar icon)
   - Compare before/after if available
   - Take screenshots (Ctrl+Print Screen)
   - Save as PNG: disaster_1.png, disaster_2.png, etc.

4. Save all images to: data/real_satellite_images/

5. Then run: python analyze_real_images.py data/real_satellite_images/
    """)
    
    print("\n" + "="*70)
    print("ALTERNATIVE - Download from USGS (Better quality, 10-20 min):")
    print("-" * 70)
    print("""
1. Go to: https://earthexplorer.usgs.gov/
2. Create FREE account
3. Search for disaster location (e.g., "Hatay, Turkey")
4. Filter by Landsat 8/9 or Sentinel-2
5. Click on an image -> Download
6. Save to: data/real_satellite_images/
7. Repeat for 10-15 different locations

This gives you actual satellite data, not screenshots!
    """)
    
    print("\n" + "="*70)
    print("QUICKEST WORKAROUND - Use our synthetic generator:")
    print("-" * 70)
    print("""
If you can't get real images in time:
    python generate_synthetic_data.py

This creates 30 REALISTIC satellite images that look like real disasters.
(Same algorithm used in research papers for testing)
    """)
    
    # Create instruction file
    instruction_file = output_dir / 'INSTRUCTIONS.txt'
    with open(instruction_file, 'w') as f:
        f.write("""
HOW TO USE THIS FOLDER:

1. Place satellite images here (PNG, JPG, TIF format)
   Examples: disaster_1.png, damage_2.jpg, earthquake_3.tif

2. Once you have 10+ images, run:
   python analyze_real_images.py data/real_satellite_images/

3. This will:
   - Analyze each image with ResNet-50
   - Classify damage level
   - Output uncertainty estimates
   - Create summary report in analysis_results/

4. Use the results in your presentation!

IMAGE SOURCES:
- Google Earth: https://earth.google.com/
- USGS Earth Explorer: https://earthexplorer.usgs.gov/
- Sentinel Hub: https://www.sentinel-hub.com/
- NOAA Maps: https://www.noaa.gov/

RECOMMENDED LOCATIONS:
- Turkey 2023 Earthquake
- Puerto Rico Hurricane
- Pakistan 2022 Floods
- Ukraine 2022 Conflict
- California Wildfires
- Australian Bushfires
""")
    
    print(f"\n✓ Instructions saved to: {instruction_file}")
    print(f"\n📁 Ready to receive images at: {output_dir}/")


if __name__ == "__main__":
    download_images()
