package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/vulkan-go/vulkan"
)

const (
	WIDTH                = 800
	HEIGHT               = 600
	MAX_FRAMES_IN_FLIGHT = 2
)

var (
	debug                  = true
	enableValidationLayers = false
	validationLayers       = []string{"VK_LAYER_KHRONOS_validation\x00"}
	deviceExtensions       = []string{"VK_KHR_swapchain\x00"}
)

func init() {
	runtime.LockOSThread()
	if debug {
		enableValidationLayers = true
	}
}

type vec2 [2]float32
type vec3 [3]float32

type Vertex struct {
	pos   vec2
	color vec3
}

func getBindingDescription() vk.VertexInputBindingDescription {
	bindingDescription := vk.VertexInputBindingDescription{
		Binding:   0,
		Stride:    uint32(unsafe.Sizeof(Vertex{})),
		InputRate: vk.VertexInputRateVertex,
	}
	return bindingDescription
}

func getAttributeDescriptions() []vk.VertexInputAttributeDescription {
	// var attributeDescriptions [2]vk.VertexInputAttributeDescription
	attributeDescriptions := make([]vk.VertexInputAttributeDescription, 2)
	attributeDescriptions[0] = vk.VertexInputAttributeDescription{
		Binding:  0,
		Location: 0,
		Format:   vk.FormatR32g32Sfloat,
		Offset:   uint32(unsafe.Offsetof(Vertex{}.pos)),
	}

	attributeDescriptions[1] = vk.VertexInputAttributeDescription{
		Binding:  0,
		Location: 1,
		Format:   vk.FormatR32g32b32Sfloat,
		Offset:   uint32(unsafe.Offsetof(Vertex{}.color)),
	}

	return attributeDescriptions
}

// var vertices = []Vertex{
// 	{vec2{0.0, -0.5}, vec3{1.0, 0.0, 0.0}},
// 	{vec2{0.5, 0.5}, vec3{0.0, 1.0, 0.0}},
// 	{vec2{-0.5, 0.5}, vec3{0.0, 0.0, 1.0}},
// }

type HelloTriangleApplication struct {
	window              *glfw.Window
	instance            vk.Instance
	debugReportCallback vk.DebugReportCallback
	surface             vk.Surface

	physicalDevice vk.PhysicalDevice
	device         vk.Device

	graphicsQueue vk.Queue
	presentQueue  vk.Queue

	swapChain             vk.Swapchain
	swapChainImages       []vk.Image
	swapChainImageFormat  vk.Format
	swapChainExtent       vk.Extent2D
	swapChainImageViews   []vk.ImageView
	swapChainFramebuffers []vk.Framebuffer

	renderPass       vk.RenderPass
	pipelineLayout   vk.PipelineLayout
	graphicsPipeline vk.Pipeline

	commandPool    vk.CommandPool
	commandBuffers []vk.CommandBuffer

	imageAvailableSemaphores []vk.Semaphore
	renderFinishedSemaphores []vk.Semaphore
	inFlightFences           []vk.Fence

	currentFrame       uint32
	framebufferResized bool
}

func (app *HelloTriangleApplication) Run() (err error) {
	if err = app.initWindow(); err != nil {
		return err
	}

	if err = app.initVulkan(); err != nil {
		return err
	}

	if err = app.mainLoop(); err != nil {
		return err
	}

	defer app.cleanup()
	return nil
}

func (app *HelloTriangleApplication) initWindow() error {
	if err := glfw.Init(); err != nil {
		return err
	}
	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	glfw.WindowHint(glfw.Resizable, glfw.False)

	window, err := glfw.CreateWindow(WIDTH, HEIGHT, "Vulkan", nil, nil)
	if err != nil {
		return err
	}
	window.SetUserPointer(window.GetUserPointer())
	window.SetFramebufferSizeCallback(func(w *glfw.Window, width int, height int) {
		app.framebufferResized = true
	})

	app.window = window
	return nil
}

func (app *HelloTriangleApplication) initVulkan() error {
	vk.SetGetInstanceProcAddr(glfw.GetVulkanGetInstanceProcAddress())
	if err := vk.Init(); err != nil {
		return err
	}

	if err := app.createInstance(); err != nil {
		return err
	}

	if err := app.setupDebugMessenger(); err != nil {
		return err
	}

	if err := app.createSurface(); err != nil {
		return err
	}

	if err := app.pickPhysicalDevice(); err != nil {
		return err
	}

	if err := app.createLogicalDevice(); err != nil {
		return err
	}

	if err := app.createSwapChain(); err != nil {
		return err
	}

	if err := app.createImageViews(); err != nil {
		return err
	}

	if err := app.createRenderPass(); err != nil {
		return err
	}

	if err := app.createGraphicsPipeline(); err != nil {
		return err
	}

	if err := app.createFramebuffers(); err != nil {
		return err
	}

	if err := app.createCommandPool(); err != nil {
		return err
	}

	if err := app.createCommandBuffers(); err != nil {
		return err
	}

	if err := app.createSyncObjects(); err != nil {
		return err
	}

	return nil
}

func (app *HelloTriangleApplication) createInstance() error {
	if enableValidationLayers && !checkValidationLayerSupport() {
		return fmt.Errorf("validation layers requested, but not available")
	}

	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "Hello Triangle",
		ApplicationVersion: vk.MakeVersion(1, 0, 0),
		PEngineName:        "No Engine",
		EngineVersion:      vk.MakeVersion(1, 0, 0),
		ApiVersion:         vk.ApiVersion10,
	}

	extensions := app.window.GetRequiredInstanceExtensions()
	if enableValidationLayers {
		extensions = append(extensions, "VK_EXT_debug_report\x00")
	}

	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &appInfo,
		EnabledExtensionCount:   uint32(len(extensions)),
		PpEnabledExtensionNames: extensions,
	}

	if enableValidationLayers {
		createInfo.PpEnabledLayerNames = validationLayers
		createInfo.EnabledLayerCount = uint32(len(validationLayers))
		dbgCreateInfo := populateDebugMessengerCreateInfo()
		createInfo.PNext = unsafe.Pointer(dbgCreateInfo.Ref())
	} else {
		createInfo.EnabledLayerCount = 0
		createInfo.PNext = nil
	}

	var instance vk.Instance
	if vk.CreateInstance(&createInfo, nil, &instance) != vk.Success {
		return fmt.Errorf("failed to create instance")
	}
	app.instance = instance

	return nil
}

func (app *HelloTriangleApplication) setupDebugMessenger() error {
	if !enableValidationLayers {
		return nil
	}
	var dbg vk.DebugReportCallback
	dbgCreateInfo := populateDebugMessengerCreateInfo()

	if vk.CreateDebugReportCallback(app.instance, dbgCreateInfo, nil, &dbg) != vk.Success {
		return fmt.Errorf("failed to set up debug messenger")
	}

	app.debugReportCallback = dbg
	return nil
}

func (app *HelloTriangleApplication) createSurface() error {
	surfaceAddr, err := app.window.CreateWindowSurface(app.instance, nil)
	if err != nil {
		return err
	}
	app.surface = vk.SurfaceFromPointer(surfaceAddr)
	return nil
}

func (app *HelloTriangleApplication) pickPhysicalDevice() error {
	var deviceCount uint32
	vk.EnumeratePhysicalDevices(app.instance, &deviceCount, nil)
	if deviceCount == 0 {
		return fmt.Errorf("failed to find GPUs with Vulkan support")
	}
	devices := make([]vk.PhysicalDevice, deviceCount)
	vk.EnumeratePhysicalDevices(app.instance, &deviceCount, devices)

	for _, device := range devices {
		if isDeviceSuitable(device, app.surface) {
			app.physicalDevice = device
			break
		}
	}

	if unsafe.Pointer(app.physicalDevice) == vk.NullHandle {
		return fmt.Errorf("failed to find a suitable gpu")
	}

	return nil
}

func (app *HelloTriangleApplication) createLogicalDevice() error {
	indices := findQueueFamilies(app.physicalDevice, app.surface)

	queueCreateInfos := []vk.DeviceQueueCreateInfo{}
	uniqueQueueFamilies := make(map[uint32]any)
	uniqueQueueFamilies[*indices.graphicsFamily] = nil
	uniqueQueueFamilies[*indices.presentFamily] = nil

	var queuePriority float32 = 1.0
	for queueFamily := range uniqueQueueFamilies {
		queueCreateInfo := vk.DeviceQueueCreateInfo{
			SType:            vk.StructureTypeDeviceQueueCreateInfo,
			QueueFamilyIndex: queueFamily,
			QueueCount:       1,
			PQueuePriorities: []float32{queuePriority},
		}
		queueCreateInfos = append(queueCreateInfos, queueCreateInfo)
	}

	deviceFeatures := []vk.PhysicalDeviceFeatures{{}}
	createInfo := vk.DeviceCreateInfo{
		SType:                   vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount:    uint32(len(queueCreateInfos)),
		PQueueCreateInfos:       queueCreateInfos,
		PEnabledFeatures:        deviceFeatures,
		EnabledExtensionCount:   uint32(len(deviceExtensions)),
		PpEnabledExtensionNames: deviceExtensions,
	}

	if enableValidationLayers {
		createInfo.EnabledLayerCount = uint32(len(validationLayers))
		createInfo.PpEnabledLayerNames = validationLayers
	} else {
		createInfo.EnabledLayerCount = 0
	}

	var device vk.Device
	if vk.CreateDevice(app.physicalDevice, &createInfo, nil, &device) != vk.Success {
		return fmt.Errorf("could not create logical device")
	}
	app.device = device

	var graphicsQueue, presentQueue vk.Queue
	vk.GetDeviceQueue(device, *indices.graphicsFamily, 0, &graphicsQueue)
	vk.GetDeviceQueue(device, *indices.presentFamily, 0, &presentQueue)
	app.graphicsQueue = graphicsQueue
	app.presentQueue = presentQueue

	return nil
}

func (app *HelloTriangleApplication) createSwapChain() error {
	swapChainSupport := querySwapChainSupport(app.physicalDevice, app.surface)
	swapChainSupport.capabilities.Deref()
	swapChainSupport.capabilities.Free()

	surfaceFormat := chooseSwapSurfaceFormat(swapChainSupport.formats)
	presentMode := chooseSwapPresentMode(swapChainSupport.presentModes)
	extent := chooseSwapExtent(swapChainSupport.capabilities, app.window)
	imageCount := swapChainSupport.capabilities.MinImageCount + 1

	if swapChainSupport.capabilities.MaxImageCount > 0 && imageCount > swapChainSupport.capabilities.MaxImageCount {
		imageCount = swapChainSupport.capabilities.MaxImageCount
	}

	createInfo := vk.SwapchainCreateInfo{
		SType:            vk.StructureTypeSwapchainCreateInfo,
		Surface:          app.surface,
		MinImageCount:    imageCount,
		ImageFormat:      surfaceFormat.Format,
		ImageColorSpace:  surfaceFormat.ColorSpace,
		ImageExtent:      extent,
		ImageArrayLayers: 1,
		ImageUsage:       vk.ImageUsageFlags(vk.ImageUsageColorAttachmentBit),
		PreTransform:     swapChainSupport.capabilities.CurrentTransform,
		CompositeAlpha:   vk.CompositeAlphaOpaqueBit,
		PresentMode:      presentMode,
		Clipped:          vk.True,
		OldSwapchain:     vk.NullSwapchain,
	}

	indices := findQueueFamilies(app.physicalDevice, app.surface)
	queueFamilyIndices := []uint32{*indices.presentFamily, *indices.graphicsFamily}

	if *indices.graphicsFamily != *indices.presentFamily {
		createInfo.ImageSharingMode = vk.SharingModeConcurrent
		createInfo.QueueFamilyIndexCount = 2
		createInfo.PQueueFamilyIndices = queueFamilyIndices
	} else {
		createInfo.ImageSharingMode = vk.SharingModeExclusive
		createInfo.QueueFamilyIndexCount = 0
		createInfo.PQueueFamilyIndices = nil
	}

	var swapChain vk.Swapchain
	if vk.CreateSwapchain(app.device, &createInfo, nil, &swapChain) != vk.Success {
		return fmt.Errorf("failed to create swap chain")
	}
	app.swapChain = swapChain

	var imagesCount uint32
	vk.GetSwapchainImages(app.device, app.swapChain, &imagesCount, nil)
	app.swapChainImages = make([]vk.Image, imageCount)
	vk.GetSwapchainImages(app.device, app.swapChain, &imagesCount, app.swapChainImages)

	app.swapChainImageFormat = surfaceFormat.Format
	app.swapChainExtent = extent
	return nil
}

func (app *HelloTriangleApplication) createImageViews() error {
	app.swapChainImageViews = make([]vk.ImageView, len(app.swapChainImages))

	for i, image := range app.swapChainImages {
		createInfo := vk.ImageViewCreateInfo{
			SType:    vk.StructureTypeImageViewCreateInfo,
			Image:    image,
			ViewType: vk.ImageViewType2d,
			Format:   app.swapChainImageFormat,
			Components: vk.ComponentMapping{
				R: vk.ComponentSwizzleIdentity,
				G: vk.ComponentSwizzleIdentity,
				B: vk.ComponentSwizzleIdentity,
				A: vk.ComponentSwizzleIdentity,
			},
			SubresourceRange: vk.ImageSubresourceRange{
				AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
				BaseMipLevel:   0,
				LevelCount:     1,
				BaseArrayLayer: 0,
				LayerCount:     1,
			},
		}
		var imageView vk.ImageView
		if vk.CreateImageView(app.device, &createInfo, nil, &imageView) != vk.Success {
			return fmt.Errorf("failed to create image views")
		}
		app.swapChainImageViews[i] = imageView
	}

	return nil
}

func (app *HelloTriangleApplication) createRenderPass() error {
	colorAttachments := []vk.AttachmentDescription{{
		Format:         app.swapChainImageFormat,
		Samples:        vk.SampleCountFlagBits(vk.SampleCount1Bit),
		LoadOp:         vk.AttachmentLoadOpClear,
		StoreOp:        vk.AttachmentStoreOpStore,
		StencilLoadOp:  vk.AttachmentLoadOpDontCare,
		StencilStoreOp: vk.AttachmentStoreOpDontCare,
		InitialLayout:  vk.ImageLayoutUndefined,
		FinalLayout:    vk.ImageLayoutPresentSrc,
	}}

	colorAttachmentRefs := []vk.AttachmentReference{{
		Attachment: 0,
		Layout:     vk.ImageLayoutColorAttachmentOptimal,
	}}

	subpasses := []vk.SubpassDescription{{
		PipelineBindPoint:    vk.PipelineBindPointGraphics,
		ColorAttachmentCount: uint32(len(colorAttachmentRefs)),
		PColorAttachments:    colorAttachmentRefs,
	}}

	renderPassCreateInfo := vk.RenderPassCreateInfo{
		SType:           vk.StructureTypeRenderPassCreateInfo,
		AttachmentCount: uint32(len(colorAttachments)),
		PAttachments:    colorAttachments,
		SubpassCount:    uint32(len(subpasses)),
		PSubpasses:      subpasses,
	}

	var renderPass vk.RenderPass

	if vk.CreateRenderPass(app.device, &renderPassCreateInfo, nil, &renderPass) != vk.Success {
		return fmt.Errorf("failed to create render pass")
	}

	app.renderPass = renderPass

	return nil
}

func (app *HelloTriangleApplication) createGraphicsPipeline() error {
	_, fileName, _, _ := runtime.Caller(0)
	dirpath := filepath.Dir(fileName)

	vertShaderCode, err := os.ReadFile(dirpath + "/shaders/vert.spv")
	if err != nil {
		return err
	}
	fragShaderCode, err := os.ReadFile(dirpath + "/shaders/frag.spv")
	if err != nil {
		return err
	}

	vertShaderModule, err := createShaderModule(vertShaderCode, app.device)
	if err != nil {
		return err
	}
	fragShaderModule, err := createShaderModule(fragShaderCode, app.device)
	if err != nil {
		return err
	}

	vertShaderStageInfo := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageVertexBit,
		Module: vertShaderModule,
		PName:  "main\x00",
	}
	fragShaderStageInfo := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageFragmentBit,
		Module: fragShaderModule,
		PName:  "main\x00",
	}

	shaderStages := []vk.PipelineShaderStageCreateInfo{vertShaderStageInfo, fragShaderStageInfo}

	bindingDescription := getBindingDescription()
	attributeDescriptions := getAttributeDescriptions()
	vertexInputInfo := vk.PipelineVertexInputStateCreateInfo{
		SType:                           vk.StructureTypePipelineVertexInputStateCreateInfo,
		VertexBindingDescriptionCount:   1,
		PVertexBindingDescriptions:      []vk.VertexInputBindingDescription{bindingDescription},
		VertexAttributeDescriptionCount: uint32(len(attributeDescriptions)),
		PVertexAttributeDescriptions:    attributeDescriptions,
	}

	inputAssembly := vk.PipelineInputAssemblyStateCreateInfo{
		SType:                  vk.StructureTypePipelineInputAssemblyStateCreateInfo,
		Topology:               vk.PrimitiveTopologyTriangleList,
		PrimitiveRestartEnable: vk.False,
	}

	viewportState := vk.PipelineViewportStateCreateInfo{
		SType:         vk.StructureTypePipelineViewportStateCreateInfo,
		ViewportCount: 1,
		ScissorCount:  1,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo{
		SType:                   vk.StructureTypePipelineRasterizationStateCreateInfo,
		DepthClampEnable:        vk.False,
		RasterizerDiscardEnable: vk.False,
		PolygonMode:             vk.PolygonModeFill,
		LineWidth:               1.0,
		CullMode:                vk.CullModeFlags(vk.CullModeBackBit),
		FrontFace:               vk.FrontFaceClockwise,
		DepthBiasEnable:         vk.False,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo{
		SType:                vk.StructureTypePipelineMultisampleStateCreateInfo,
		SampleShadingEnable:  vk.False,
		RasterizationSamples: vk.SampleCount1Bit,
	}

	colorBlendAttachments := []vk.PipelineColorBlendAttachmentState{{
		ColorWriteMask: vk.ColorComponentFlags(vk.ColorComponentRBit | vk.ColorComponentGBit | vk.ColorComponentBBit | vk.ColorComponentABit),
		BlendEnable:    vk.False,
	}}

	colorBlending := vk.PipelineColorBlendStateCreateInfo{
		SType:           vk.StructureTypePipelineColorBlendStateCreateInfo,
		LogicOpEnable:   vk.False,
		LogicOp:         vk.LogicOpCopy,
		AttachmentCount: 1,
		PAttachments:    colorBlendAttachments,
		BlendConstants:  [4]float32{0.0, 0.0, 0.0, 0.0},
	}

	dynamicStates := []vk.DynamicState{
		vk.DynamicStateViewport,
		vk.DynamicStateScissor,
	}

	dynamicState := vk.PipelineDynamicStateCreateInfo{
		SType:             vk.StructureTypePipelineDynamicStateCreateInfo,
		DynamicStateCount: uint32(len(dynamicStates)),
		PDynamicStates:    dynamicStates,
	}

	pipelineLayoutInfo := vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         0,
		PushConstantRangeCount: 0,
	}

	var pipelineLayout vk.PipelineLayout
	if vk.CreatePipelineLayout(app.device, &pipelineLayoutInfo, nil, &pipelineLayout) != vk.Success {
		return fmt.Errorf("failed to create pipeline layout")
	}
	app.pipelineLayout = pipelineLayout

	pipelineInfos := []vk.GraphicsPipelineCreateInfo{{
		SType:               vk.StructureTypeGraphicsPipelineCreateInfo,
		StageCount:          2,
		PStages:             shaderStages,
		PVertexInputState:   &vertexInputInfo,
		PInputAssemblyState: &inputAssembly,
		PViewportState:      &viewportState,
		PRasterizationState: &rasterizer,
		PMultisampleState:   &multisampling,
		PColorBlendState:    &colorBlending,
		PDynamicState:       &dynamicState,
		Layout:              app.pipelineLayout,
		RenderPass:          app.renderPass,
		Subpass:             0,
		BasePipelineHandle:  vk.Pipeline(vk.NullHandle),
	}}

	var graphicsPipelines = make([]vk.Pipeline, 1)

	if vk.CreateGraphicsPipelines(app.device, vk.PipelineCache(vk.NullHandle), uint32(len(pipelineInfos)), pipelineInfos, nil, graphicsPipelines) != vk.Success {
		return fmt.Errorf("failed to create graphics pipeline")
	}

	app.graphicsPipeline = graphicsPipelines[0]

	vk.DestroyShaderModule(app.device, fragShaderModule, nil)
	vk.DestroyShaderModule(app.device, vertShaderModule, nil)
	return nil
}

func (app *HelloTriangleApplication) createFramebuffers() error {
	app.swapChainFramebuffers = make([]vk.Framebuffer, len(app.swapChainImageViews))

	for i := range app.swapChainImageViews {
		attachments := []vk.ImageView{
			app.swapChainImageViews[i],
		}

		fbCreateInfo := vk.FramebufferCreateInfo{
			SType:           vk.StructureTypeFramebufferCreateInfo,
			RenderPass:      app.renderPass,
			AttachmentCount: uint32(len(attachments)),
			PAttachments:    attachments,
			Width:           app.swapChainExtent.Width,
			Height:          app.swapChainExtent.Height,
			Layers:          1,
		}

		var framebuffer vk.Framebuffer
		if vk.CreateFramebuffer(app.device, &fbCreateInfo, nil, &framebuffer) != vk.Success {
			return fmt.Errorf("failed to create framebuffer")
		}

		app.swapChainFramebuffers[i] = framebuffer
	}

	return nil
}

func (app *HelloTriangleApplication) createCommandPool() error {
	queueFamilyIndices := findQueueFamilies(app.physicalDevice, app.surface)

	poolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		Flags:            vk.CommandPoolCreateFlags(vk.CommandPoolCreateResetCommandBufferBit),
		QueueFamilyIndex: *queueFamilyIndices.graphicsFamily,
	}
	var commandPool vk.CommandPool
	if vk.CreateCommandPool(app.device, &poolInfo, nil, &commandPool) != vk.Success {
		return fmt.Errorf("failed to create command pool")
	}
	app.commandPool = commandPool
	return nil
}

func (app *HelloTriangleApplication) createCommandBuffers() error {
	commandBuffers := make([]vk.CommandBuffer, MAX_FRAMES_IN_FLIGHT)

	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        app.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: uint32(len(commandBuffers)),
	}

	if vk.AllocateCommandBuffers(app.device, &allocInfo, commandBuffers) != vk.Success {
		return fmt.Errorf("failed to allocate command buffers")
	}
	app.commandBuffers = commandBuffers
	return nil
}

func (app *HelloTriangleApplication) createSyncObjects() error {
	semaphoreInfo := vk.SemaphoreCreateInfo{
		SType: vk.StructureTypeSemaphoreCreateInfo,
	}

	fenceInfo := vk.FenceCreateInfo{
		SType: vk.StructureTypeFenceCreateInfo,
		Flags: vk.FenceCreateFlags(vk.FenceCreateSignaledBit),
	}

	app.imageAvailableSemaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT)
	app.renderFinishedSemaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT)
	app.inFlightFences = make([]vk.Fence, MAX_FRAMES_IN_FLIGHT)
	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i++ {
		var imageAvailableSemaphore, renderFinishedSemaphore vk.Semaphore
		var inFlightFence vk.Fence
		if vk.CreateSemaphore(app.device, &semaphoreInfo, nil, &imageAvailableSemaphore) != vk.Success ||
			vk.CreateSemaphore(app.device, &semaphoreInfo, nil, &renderFinishedSemaphore) != vk.Success ||
			vk.CreateFence(app.device, &fenceInfo, nil, &inFlightFence) != vk.Success {
			return fmt.Errorf("failed to create synchronization objects for a frame")
		}

		app.imageAvailableSemaphores[i] = imageAvailableSemaphore
		app.renderFinishedSemaphores[i] = renderFinishedSemaphore
		app.inFlightFences[i] = inFlightFence
	}

	return nil
}

func (app *HelloTriangleApplication) mainLoop() error {
	for !app.window.ShouldClose() {
		glfw.PollEvents()
		if err := app.drawFrame(); err != nil {
			return err
		}
	}

	vk.DeviceWaitIdle(app.device)
	return nil
}

func (app *HelloTriangleApplication) drawFrame() error {
	vk.WaitForFences(app.device, 1, []vk.Fence{app.inFlightFences[app.currentFrame]}, vk.True, vk.MaxUint64)

	var imageIndex uint32
	result := vk.AcquireNextImage(app.device, app.swapChain, vk.MaxUint64, app.imageAvailableSemaphores[app.currentFrame], vk.Fence(vk.NullHandle), &imageIndex)

	if result == vk.ErrorOutOfDate {
		return app.recreateSwapChain()
	} else if result != vk.Success && result != vk.Suboptimal {
		return fmt.Errorf("failed to acquire swap chain image")
	}

	vk.ResetFences(app.device, 1, []vk.Fence{app.inFlightFences[app.currentFrame]})

	vk.ResetCommandBuffer(app.commandBuffers[app.currentFrame], 0)
	app.recordCommandBuffer(app.commandBuffers[app.currentFrame], imageIndex)

	waitSemaphores := []vk.Semaphore{app.imageAvailableSemaphores[app.currentFrame]}
	waitStages := []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)}

	commandBuffers := []vk.CommandBuffer{app.commandBuffers[app.currentFrame]}
	signalSemaphores := []vk.Semaphore{app.renderFinishedSemaphores[app.currentFrame]}

	submitInfos := []vk.SubmitInfo{{
		SType:                vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount:   uint32(len(waitSemaphores)),
		PWaitSemaphores:      waitSemaphores,
		PWaitDstStageMask:    waitStages,
		CommandBufferCount:   uint32(len(commandBuffers)),
		PCommandBuffers:      commandBuffers,
		SignalSemaphoreCount: uint32(len(signalSemaphores)),
		PSignalSemaphores:    signalSemaphores,
	}}

	if vk.QueueSubmit(app.graphicsQueue, 1, submitInfos, app.inFlightFences[app.currentFrame]) != vk.Success {
		return fmt.Errorf("failed to submit draw command buffer")
	}

	swapChains := []vk.Swapchain{app.swapChain}
	presentInfo := vk.PresentInfo{
		SType:              vk.StructureTypePresentInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    signalSemaphores,
		SwapchainCount:     uint32(len(swapChains)),
		PSwapchains:        swapChains,
		PImageIndices:      []uint32{imageIndex},
	}

	result = vk.QueuePresent(app.presentQueue, &presentInfo)
	if result == vk.ErrorOutOfDate || result == vk.Suboptimal || app.framebufferResized {
		app.framebufferResized = false
		return app.recreateSwapChain()
	} else if result != vk.Success {
		return fmt.Errorf("failed to present swap chain image")
	}

	app.currentFrame = (app.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT
	return nil
}

func (app *HelloTriangleApplication) recreateSwapChain() error {
	width, height := app.window.GetFramebufferSize()
	for width == 0 || height == 0 {
		width, height = app.window.GetFramebufferSize()
		glfw.WaitEvents()
	}
	vk.DeviceWaitIdle(app.device)
	app.cleanupSwapChain()

	if err := app.createSwapChain(); err != nil {
		return err
	}

	if err := app.createImageViews(); err != nil {
		return err
	}

	if err := app.createFramebuffers(); err != nil {
		return err
	}

	return nil
}

func (app *HelloTriangleApplication) cleanupSwapChain() {
	for _, framebuffer := range app.swapChainFramebuffers {
		vk.DestroyFramebuffer(app.device, framebuffer, nil)
	}

	for i := range app.swapChainImageViews {
		vk.DestroyImageView(app.device, app.swapChainImageViews[i], nil)
	}

	vk.DestroySwapchain(app.device, app.swapChain, nil)
}

func (app *HelloTriangleApplication) cleanup() {
	app.cleanupSwapChain()

	vk.DestroyPipeline(app.device, app.graphicsPipeline, nil)
	vk.DestroyPipelineLayout(app.device, app.pipelineLayout, nil)

	vk.DestroyRenderPass(app.device, app.renderPass, nil)

	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i++ {
		vk.DestroySemaphore(app.device, app.renderFinishedSemaphores[i], nil)
		vk.DestroySemaphore(app.device, app.imageAvailableSemaphores[i], nil)
		vk.DestroyFence(app.device, app.inFlightFences[i], nil)
	}

	vk.DestroyCommandPool(app.device, app.commandPool, nil)

	vk.DestroyDevice(app.device, nil)

	if enableValidationLayers {
		vk.DestroyDebugReportCallback(app.instance, app.debugReportCallback, nil)
	}

	vk.DestroySurface(app.instance, app.surface, nil)
	vk.DestroyInstance(app.instance, nil)

	app.window.Destroy()
	glfw.Terminate()
}

func (app *HelloTriangleApplication) recordCommandBuffer(commandBuffer vk.CommandBuffer, imageIndex uint32) error {
	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}

	if vk.BeginCommandBuffer(commandBuffer, &beginInfo) != vk.Success {
		return fmt.Errorf("failed to begin recording command buffer")
	}

	var clearColor vk.ClearValue
	clearColor.SetColor([]float32{0, 0, 0, 1})

	renderPassInfo := vk.RenderPassBeginInfo{
		SType:       vk.StructureTypeRenderPassBeginInfo,
		RenderPass:  app.renderPass,
		Framebuffer: app.swapChainFramebuffers[imageIndex],
		RenderArea: vk.Rect2D{
			Offset: vk.Offset2D{
				X: 0, Y: 0,
			},
			Extent: app.swapChainExtent,
		},
		ClearValueCount: 1,
		PClearValues:    []vk.ClearValue{clearColor},
	}

	vk.CmdBeginRenderPass(commandBuffer, &renderPassInfo, vk.SubpassContentsInline)
	vk.CmdBindPipeline(commandBuffer, vk.PipelineBindPointGraphics, app.graphicsPipeline)

	viewports := []vk.Viewport{{
		X:        0.0,
		Y:        0.0,
		Width:    float32(app.swapChainExtent.Width),
		Height:   float32(app.swapChainExtent.Height),
		MinDepth: 0.0,
		MaxDepth: 1.0,
	}}
	vk.CmdSetViewport(commandBuffer, 0, 1, viewports)

	scissors := []vk.Rect2D{{
		Offset: vk.Offset2D{X: 0, Y: 0},
		Extent: app.swapChainExtent,
	}}
	vk.CmdSetScissor(commandBuffer, 0, 1, scissors)

	vk.CmdDraw(commandBuffer, 3, 1, 0, 0)
	vk.CmdEndRenderPass(commandBuffer)

	if vk.EndCommandBuffer(commandBuffer) != vk.Success {
		return fmt.Errorf("failed to record command buffer")
	}
	return nil
}

func createShaderModule(code []byte, device vk.Device) (vk.ShaderModule, error) {
	createInfo := vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(code)),
		PCode:    bytesToBytecode(code),
	}

	var shaderModule vk.ShaderModule
	err := vk.Error(vk.CreateShaderModule(device, &createInfo, nil, &shaderModule))
	if err != nil {
		return nil, fmt.Errorf("could not create shader module - " + err.Error())
	}
	return shaderModule, nil
}

type swapChainSupportDetails struct {
	capabilities vk.SurfaceCapabilities
	formats      []vk.SurfaceFormat
	presentModes []vk.PresentMode
}

func chooseSwapExtent(capabilities vk.SurfaceCapabilities, window *glfw.Window) vk.Extent2D {
	// capabilities.Deref()
	// capabilities.Free()
	// capabilities.CurrentExtent.Deref()
	// capabilities.CurrentExtent.Free()
	capabilities.MaxImageExtent.Deref()
	capabilities.MaxImageExtent.Free()
	capabilities.MinImageExtent.Deref()
	capabilities.MinImageExtent.Free()

	// if capabilities.CurrentExtent.Width != vk.MaxUint32 { // ?? imageExtent must be between minImageExtent and maxImageExtent
	// 	return capabilities.CurrentExtent
	// }

	width, height := window.GetFramebufferSize()
	actualExtent := vk.Extent2D{
		Width:  uint32(width),
		Height: uint32(height),
	}

	actualExtent.Width = clamp(actualExtent.Width, capabilities.MinImageExtent.Width, capabilities.MaxImageExtent.Width)
	actualExtent.Height = clamp(actualExtent.Height, capabilities.MinImageExtent.Height, capabilities.MaxImageExtent.Height)
	return actualExtent
}

func chooseSwapPresentMode(availablePresentModes []vk.PresentMode) vk.PresentMode {
	for _, availablePresentMode := range availablePresentModes {
		if availablePresentMode == vk.PresentModeMailbox {
			return availablePresentMode
		}
	}

	return vk.PresentModeFifo
}

func chooseSwapSurfaceFormat(availableFormats []vk.SurfaceFormat) vk.SurfaceFormat {
	if len(availableFormats) < 1 {
		return vk.SurfaceFormat{}
	}

	for _, availableFormat := range availableFormats {
		availableFormat.Deref()
		if availableFormat.Format == vk.FormatB8g8r8a8Srgb && availableFormat.ColorSpace == vk.ColorspaceSrgbNonlinear {
			return availableFormat
		}
		availableFormat.Free()
	}

	return availableFormats[0]
}

func querySwapChainSupport(device vk.PhysicalDevice, surface vk.Surface) swapChainSupportDetails {
	var details swapChainSupportDetails
	vk.GetPhysicalDeviceSurfaceCapabilities(device, surface, &details.capabilities)

	var formatCount uint32
	vk.GetPhysicalDeviceSurfaceFormats(device, surface, &formatCount, nil)
	if formatCount != 0 {
		formats := make([]vk.SurfaceFormat, formatCount)
		vk.GetPhysicalDeviceSurfaceFormats(device, surface, &formatCount, formats)
		details.formats = formats
	}

	var presentModeCount uint32
	vk.GetPhysicalDeviceSurfacePresentModes(device, surface, &presentModeCount, nil)
	if presentModeCount != 0 {
		presentModes := make([]vk.PresentMode, presentModeCount)
		vk.GetPhysicalDeviceSurfacePresentModes(device, surface, &presentModeCount, presentModes)
		details.presentModes = presentModes
	}
	return details
}

func checkDeviceExtensionSupport(device vk.PhysicalDevice) bool {
	var extensionCount uint32
	vk.EnumerateDeviceExtensionProperties(device, "", &extensionCount, nil)
	availableExtensions := make([]vk.ExtensionProperties, extensionCount)
	vk.EnumerateDeviceExtensionProperties(device, "", &extensionCount, availableExtensions)

	requiredExtensions := make(map[string]bool)
	for _, deviceExtension := range deviceExtensions {
		requiredExtensions[strings.Trim(deviceExtension, "\x00")] = true
	}

	for _, availableExtension := range availableExtensions {
		availableExtension.Deref()
		name := vk.ToString(availableExtension.ExtensionName[:])
		_, ok := requiredExtensions[name]
		if ok {
			delete(requiredExtensions, name)
		}
		availableExtension.Free()
	}

	return len(requiredExtensions) == 0
}

func isDeviceSuitable(device vk.PhysicalDevice, surface vk.Surface) bool {
	indices := findQueueFamilies(device, surface)

	extensionsSupported := checkDeviceExtensionSupport(device)

	var swapChainAdequate bool
	if extensionsSupported {
		swapChainSupport := querySwapChainSupport(device, surface)
		swapChainAdequate = len(swapChainSupport.formats) != 0 && len(swapChainSupport.presentModes) != 0
	}

	return indices.isComplete() && extensionsSupported && swapChainAdequate
}

type QueueFamilyIndices struct {
	graphicsFamily *uint32
	presentFamily  *uint32
}

func (q *QueueFamilyIndices) isComplete() bool {
	return q.graphicsFamily != nil && q.presentFamily != nil
}

func findQueueFamilies(device vk.PhysicalDevice, surface vk.Surface) QueueFamilyIndices {
	var indices QueueFamilyIndices
	var queueFamilyCount uint32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nil)
	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies)

	for i, queueFamily := range queueFamilies {
		queueFamily.Deref()
		queueFlags := queueFamily.QueueFlags
		queueFamily.Free()

		if queueFlags&vk.QueueFlags(vk.QueueGraphicsBit) != 0 {
			tmp := uint32(i)
			indices.graphicsFamily = &tmp
		}

		var presentSupport vk.Bool32
		vk.GetPhysicalDeviceSurfaceSupport(device, uint32(i), surface, &presentSupport)
		if presentSupport == vk.True {
			tmp := uint32(i)
			indices.presentFamily = &tmp
		}

		if indices.isComplete() {
			break
		}
	}
	return indices
}

func populateDebugMessengerCreateInfo() *vk.DebugReportCallbackCreateInfo {
	return &vk.DebugReportCallbackCreateInfo{
		SType:       vk.StructureTypeDebugReportCallbackCreateInfo,
		Flags:       vk.DebugReportFlags(vk.DebugReportPerformanceWarningBit | vk.DebugReportWarningBit | vk.DebugReportErrorBit),
		PfnCallback: debugCallback,
		PNext:       nil,
		PUserData:   nil,
	}
}

func debugCallback(flags vk.DebugReportFlags, objectType vk.DebugReportObjectType, object uint64, location uint,
	messageCode int32, layerPrefix string, message string, userData unsafe.Pointer) vk.Bool32 {
	log.Printf("validation layer: %s\n", message)
	return vk.False
}

func checkValidationLayerSupport() bool {
	var layerCount uint32
	vk.EnumerateInstanceLayerProperties(&layerCount, nil)
	availableLayers := make([]vk.LayerProperties, layerCount)
	vk.EnumerateInstanceLayerProperties(&layerCount, availableLayers)

	for _, layerName := range validationLayers {
		layerFound := false

		for _, layerProperties := range availableLayers {
			layerProperties.Deref()
			name := vk.ToString(layerProperties.LayerName[:])
			layerProperties.Free()

			if strings.Trim(layerName, "\x00") == name {
				layerFound = true
				break
			}
		}

		if !layerFound {
			return false
		}
	}
	return true
}

func main() {
	app := &HelloTriangleApplication{}

	err := app.Run()
	if err != nil {
		log.Fatalf("%+v\n", err)
	}
	// fmt.Println(getAttributeDescriptions())
	// fmt.Println(getBindingDescription())

}

func clamp[T uint32](val, min, max T) T {
	if val < min {
		return min
	} else if val > max {
		return max
	}
	return val
}

func bytesToBytecode(b []byte) []uint32 {
	byteCode := make([]uint32, len(b)/4)
	for i := 0; i < len(byteCode); i++ {
		byteIndex := i * 4
		byteCode[i] = 0
		byteCode[i] |= uint32(b[byteIndex])
		byteCode[i] |= uint32(b[byteIndex+1]) << 8
		byteCode[i] |= uint32(b[byteIndex+2]) << 16
		byteCode[i] |= uint32(b[byteIndex+3]) << 24
	}

	return byteCode
}
