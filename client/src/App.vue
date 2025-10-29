<script setup lang="ts">
import ContentItem from './components/ContentItem.vue';
import { ref } from 'vue'
const isStreaming = ref(false)
const images = ref<string[]>([]);
const selectedFile = ref<File | null>(null)
// const selectView = ref<string>()
const errorMessage = ref('')

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  processFile(file)
}


function handleDrop(event: DragEvent) {
  event.preventDefault()
  const file = event.dataTransfer?.files?.[0]
  processFile(file)
}


function processFile(file?: File) {
  if (!file) return

  // Check extension strictly, we want only .DCM file.
  // BUT need to CHECK AGAIN in BACKEND
  if (!file.name.toLowerCase().endsWith('.dcm')) {
    errorMessage.value = 'Only .dcm files are allowed.'
    selectedFile.value = null
    return
  }

  // Success
  errorMessage.value = ''
  selectedFile.value = file
}


function handleDragOver(event: DragEvent) {
  event.preventDefault()
}
const source: never[] = [];
async function submitForm(e: Event) {
  e.preventDefault();
  if (!selectedFile.value) return;

  isStreaming.value = true;
  images.value = []; 
  const formData = new FormData();
  formData.append("file", selectedFile.value);

  const response = await fetch("http://localhost:8000/api/v1/detect", {
    method: "POST",
    body: formData,
  });

  if (!response.body) {
    errorMessage.value = "No response stream.";
    isStreaming.value = false;
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  async function readChunk() {
    const { done, value } = await reader.read();
    if (done) {
      isStreaming.value = false;
      return;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      const data = JSON.parse(line);
      const imgSrc = `data:image/png;base64,${data.image}`;
      images.value.push(imgSrc);
    }

    await readChunk();
  }

  await readChunk();
}
function clearImage() {
  isStreaming.value = false;
  images.value = []
  selectedFile.value = null
}

</script>

<template>
  <!-- <aside class="w-[270px] bg-[#3b3fa2] h-screen fixed z-10"> -->
  <!-- <nav class="border-b-2 border-gray-500">
    <div class="flex justify-center items-center w-full py-2">
      <img src="/logo.svg" class="w-40 h-auto" />
    </div>
    <div>
      <ul>
        <li>

        </li>
      </ul>
    </div>
    <div>

    </div>
  </nav> -->
  <nav class="border-b-2 py-2 border-gray-500 flex justify-between items-center">
    <div class=" px-3 w-full">
      <img src="/logo.svg" class="w-40 h-auto" />
    </div>
    <div class="w-1/3">
      <ul class="flex justify-around italic gap-4 ">
        <li>
          Home
        </li>
        <li class="font-semibold underline">
          Analyse
        </li>
        <li>
          History
        </li>
        <li>
          About
        </li>
      </ul>
    </div>
    <div class="px-3 w-full">
      <div class="flex flex-row-reverse">
        <svg xmlns="http://www.w3.org/2000/svg" width="3rem" height="2.5rem" viewBox="0 -0.5 25 25" fill="none">
          <path fill-rule="evenodd" clip-rule="evenodd"
            d="M3.5 7V17C3.5 18.1046 4.39543 19 5.5 19H19.5C20.6046 19 21.5 18.1046 21.5 17V7C21.5 5.89543 20.6046 5 19.5 5H5.5C4.39543 5 3.5 5.89543 3.5 7Z"
            stroke="#000000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
          <path d="M15.5 10H18.5" stroke="#000000" stroke-width="1.5" stroke-linecap="round" />
          <path d="M15.5 13H18.5" stroke="#000000" stroke-width="1.5" stroke-linecap="round" />
          <path fill-rule="evenodd" clip-rule="evenodd"
            d="M11.5 10C11.5 11.1046 10.6046 12 9.5 12C8.39543 12 7.5 11.1046 7.5 10C7.5 8.89543 8.39543 8 9.5 8C10.0304 8 10.5391 8.21071 10.9142 8.58579C11.2893 8.96086 11.5 9.46957 11.5 10Z"
            stroke="#000000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
          <path d="M5.5 16C8.283 12.863 11.552 13.849 13.5 16" stroke="#000000" stroke-width="1.5"
            stroke-linecap="round" />
        </svg>
      </div>
    </div>
  </nav>
  <!-- <div class="flex flex-col">
      <PatientItem v-for="value in source" :key="value" :value="value" />
    </div> -->
  <!-- </aside> -->

  <div class="w-full  min-h-screen bg-gray-100">
    <div class="px-2 py-1">
      <div v-if="source.length > 0">
        <ContentItem />
      </div>
      <div v-else class="flex justify-center items-center min-w-full min-h-[70vh]">
        <form @submit.prevent="submitForm" enctype="multipart/form-data" class="w-[70%] h-full">
          <div class="rounded-4xl border-gray-600 border-2">
            <div class="border-b border-b-gray-400 px-7 py-4 flex gap-2">
              <div class="border-2 border-gray-300 rounded-full flex justify-center items-center w-16 h-16">
                <svg class="w-10 h-10 text-gray-500 dark:text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none"
                  viewBox="0 0 20 16">
                  <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                </svg>
              </div>
              <div class="flex flex-col py-1">
                <div class="font-semibold text-lg">Upload File</div>
                <div class="text-sm text-gray-400">
                  <p>Select and upload the file that you want to analyze</p>
                </div>
              </div>
            </div>

            <div class="flex items-center justify-center w-full p-3">
              <label for="dropzone-file" v-if="!selectedFile"
                class="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-[20px] cursor-pointer bg-gray-50 hover:bg-gray-100"
                @drop="handleDrop" @dragover="handleDragOver">

                <div class="flex flex-col items-center justify-center pt-5 pb-6">
                  <svg class="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400" xmlns="http://www.w3.org/2000/svg"
                    fill="none" viewBox="0 0 20 16">
                    <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                  </svg>
                  <p class="mb-2 text-sm text-gray-500">
                    <span class="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p class="text-xs text-gray-500">DCM (Only Accept 1 file)</p>
                </div>
                <input id="dropzone-file" type="file" class="hidden" accept=".dcm" @change="handleFileSelect" />
              </label>
            </div>

            <div v-if="selectedFile" class="pl-7 pr-7 font-semibold pb-3">
              <div class="border p-2 rounded-[20px] flex items-center gap-3 bg-gray-50 border-gray-500">
                <svg xmlns="http://www.w3.org/2000/svg" width="2rem" height="2rem" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M19 9V17.8C19 18.9201 19 19.4802 18.782 19.908C18.5903 20.2843 18.2843 20.5903 17.908 20.782C17.4802 21 16.9201 21 15.8 21H8.2C7.07989 21 6.51984 21 6.09202 20.782C5.71569 20.5903 5.40973 20.2843 5.21799 19.908C5 19.4802 5 18.9201 5 17.8V6.2C5 5.07989 5 4.51984 5.21799 4.09202C5.40973 3.71569 5.71569 3.40973 6.09202 3.21799C6.51984 3 7.0799 3 8.2 3H13M19 9L13 3M19 9H14C13.4477 9 13 8.55228 13 8V3"
                    stroke="#6a7282" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
                <span class="truncate overflow-hidden">{{ selectedFile.name }}</span>
                <button @click="selectedFile = null"
                  class="ml-auto hover:cursor-pointer text-gray-500 hover:text-red-600">✕</button>
              </div>
            </div>

            <div v-if="errorMessage" class="text-center text-red-600 font-semibold pb-3">
              {{ errorMessage }}
            </div>
          </div>

          <div class="flex justify-center">
            <button type="submit"
              class="bg-[#4664AD] disabled:bg-[#707fab] text-white mt-4 font-semibold border border-transparent ease-in hover:bg-[#0F1B43] transition rounded-xl px-5 py-3"
              :disabled="!selectedFile">
              SUBMIT
            </button>
          </div>
        </form>


      </div>
      <div v-if="images.length > 0" class="mt-5 grid grid-cols-3 gap-4 p-4 bg-gray-100 rounded-lg shadow-md">
        <div v-for="(img, idx) in images" :key="idx" class="border rounded-lg overflow-hidden bg-white">
          <img :src="img" class="w-full h-auto object-contain" />
        </div>
      </div>

      <div v-if="isStreaming" class="text-gray-500 mt-2 text-center">Streaming frames...</div>
      <div class="pb-48 pt-4 flex justify-center w-full">
        <button v-if="images.length > 0" class="font-semibold text-xl bg-blue-300 px-5 hover:cursor-pointer hover:bg-blue-800 hover:text-white transition ease-out  py-2 rounded-lg  text-center " @click="clearImage">
          CLEAR
        </button>
      </div>
    </div>
  </div>
</template>
