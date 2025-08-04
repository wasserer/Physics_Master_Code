import os

folderPath = ''
dat_files = [f for f in os.listdir(folderPath) if f.endswith('.dat')]\
for 



def importData(self):
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]
        for fileName in self.dat_files:
            filePath = os.path.join(self.folderPath, fileName)
            data = np.loadtxt(filePath, skiprows=21)
            wavelength = data[:, 1].tolist()
            value = data[:, 2].tolist()
            #Label: dark or light
            fnameLower = fileName.lower()
            if "dark" in fnameLower:
                label = "Dark"
            elif "light" in fnameLower:
                label = "Light"
            else:
                label = "Unknown"
            #Import timestamp:
            with open(filePath, 'r') as f:
                lines = f.readlines()
                first_line = lines[0].strip()
                temp_line = lines[5].strip()
                timestampStr = first_line.split(":", 1)[1].strip()
                timestamp = datetime.strptime(timestampStr, "%Y-%m-%d %H:%M:%S")
            #Import temperature:
                temp_str = temp_line.split(":")[1].strip().split()[0]
                temperature = float(temp_str)
            #Log the peak position, height and the FWHM:(editing)

                # Convert to NumPy array for slicing
                wavelength = np.array(wavelength)
                value = np.array(value)

                # Find peak
                max_index = np.argmax(value)
                peak_position = wavelength[max_index]
                peak_height = value[max_index]
                half_max = peak_height / 2

                # Find left half max index (closest from the left)
                left_indices = np.where(value[:max_index] < half_max)[0]
                if left_indices.size > 0:
                    left_index = left_indices[-1]
                    # Linear interpolation
                    x1, x2 = wavelength[left_index], wavelength[left_index + 1]
                    y1, y2 = value[left_index], value[left_index + 1]
                    left_half = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                else:
                    left_half = wavelength[0]

                # Find right half max index (closest from the right)
                right_indices = np.where(value[max_index:] < half_max)[0]
                if right_indices.size > 0:
                    right_index = right_indices[0] + max_index
                    # Linear interpolation
                    x1, x2 = wavelength[right_index - 1], wavelength[right_index]
                    y1, y2 = value[right_index - 1], value[right_index]
                    right_half = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                else:
                    right_half = wavelength[-1]

                fwhm = right_half - left_half

            #Load the data in the module:
            self.wavelengths.append(wavelength)
            self.values.append(value)
            self.temperatures.append(temperature)
            self.labels.append(label)
            self.timestamps.append(timestamp)
            self.PeakHeight.append(peak_height)
            self.PeakPosition.append(peak_position)
            self.PeakFWHM.append(fwhm)