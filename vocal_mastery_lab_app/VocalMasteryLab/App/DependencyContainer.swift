import Foundation
import VocalisDomain
import SubscriptionDomain

@MainActor
public class DependencyContainer {
    public static let shared = DependencyContainer()

    private init() {
        // Initialize dependencies
        setupInfrastructure()
        setupUseCases()
    }

    // MARK: - Infrastructure Layer

    private lazy var logger: LoggerProtocol = {
        OSLogAdapter.useCase
    }()

    private lazy var audioRecorder: AudioRecorderProtocol = {
        AVAudioRecorderWrapper()
    }()

    public lazy var audioPlayer: AudioPlayerProtocol = {
        AVAudioPlayerWrapper(settingsRepository: audioSettingsRepository)
    }()

    public lazy var recordingRepository: RecordingRepositoryProtocol = {
        FileRecordingRepository(pitchDataCache: pitchDataCache)
    }()

    public lazy var pitchDetector: RealtimePitchDetector = {
        // Load audio detection settings from repository
        let settings = audioSettingsRepository.get()

        return RealtimePitchDetector(
            rmsSilenceThreshold: settings.rmsSilenceThreshold,
            confidenceThreshold: settings.confidenceThreshold
        )
    }()

    /// Factory for creating audio file analyzers with current settings
    /// Each call creates a new analyzer with the currently configured pitch algorithm
    private lazy var audioFileAnalyzerFactory: AudioFileAnalyzerFactoryProtocol = {
        AudioFileAnalyzerFactory(settingsRepository: audioSettingsRepository)
    }()

    private lazy var analysisCache: AnalysisCacheProtocol = {
        AnalysisCache(maxCacheSize: 10)
    }()

    private lazy var pitchDataCache: PitchDataCacheProtocol = {
        FilePitchDataCache()
    }()

    // Subscription Infrastructure
    private lazy var storeKitProductService: StoreKitProductServiceProtocol = {
        StoreKitProductService()
    }()

    private lazy var storeKitPurchaseService: StoreKitPurchaseServiceProtocol = {
        StoreKitPurchaseService()
    }()

    private lazy var userCohortStore: UserCohortStoreProtocol = {
        UserDefaultsCohortStore()
    }()

    public lazy var subscriptionRepository: SubscriptionRepositoryProtocol = {
        StoreKitSubscriptionRepository(
            productService: storeKitProductService,
            purchaseService: storeKitPurchaseService,
            cohortStore: userCohortStore
        )
    }()

    public lazy var audioSettingsRepository: AudioSettingsRepositoryProtocol = {
        UserDefaultsAudioSettingsRepository()
    }()

    public lazy var extractedAudioRepository: ExtractedAudioRepositoryProtocol = {
        FileExtractedAudioRepository()
    }()

    public lazy var vocalExtractor: VocalExtractorProtocol = {
        // Try to load CoreML model from bundle (check both package and compiled formats)
        if let modelURL = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") {
            return CoreMLVocalExtractor(modelURL: modelURL)
        }
        // Try compiled model format
        if let modelURL = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlmodelc") {
            return CoreMLVocalExtractor(modelURL: modelURL)
        }
        // Fallback to mock if model not found
        return MockVocalExtractor()
    }()

    // MARK: - Application Layer

    // Domain Services
    private lazy var recordingPolicyService: RecordingPolicyService = {
        RecordingPolicyServiceImpl()
    }()

    private lazy var startRecordingUseCase: StartRecordingUseCaseProtocol = {
        StartRecordingUseCase(
            audioRecorder: audioRecorder,
            recordingPolicyService: recordingPolicyService
        )
    }()

    private lazy var stopRecordingUseCase: StopRecordingUseCaseProtocol = {
        StopRecordingUseCase(
            audioRecorder: audioRecorder,
            recordingRepository: recordingRepository
        )
    }()

    public lazy var analyzeRecordingUseCase: AnalyzeRecordingUseCase = {
        AnalyzeRecordingUseCase(
            analyzerFactory: audioFileAnalyzerFactory,
            analysisCache: analysisCache,
            pitchDataCache: pitchDataCache,
            audioSettingsRepository: audioSettingsRepository,
            recordingRepository: recordingRepository,
            logger: logger
        )
    }()

    // Subscription Use Cases
    private lazy var getSubscriptionStatusUseCase: GetSubscriptionStatusUseCaseProtocol = {
        GetSubscriptionStatusUseCase(repository: subscriptionRepository)
    }()

    private lazy var purchaseSubscriptionUseCase: PurchaseSubscriptionUseCaseProtocol = {
        PurchaseSubscriptionUseCase(repository: subscriptionRepository)
    }()

    private lazy var restorePurchasesUseCase: RestorePurchasesUseCaseProtocol = {
        RestorePurchasesUseCase(repository: subscriptionRepository)
    }()

    private lazy var getAvailableProductsUseCase: GetAvailableProductsUseCaseProtocol = {
        GetAvailableProductsUseCase(repository: subscriptionRepository)
    }()

    // MARK: - Presentation Layer

    public lazy var recordingViewModel: RecordingViewModel = {
        #if DEBUG
        let disableCountdown = CommandLine.arguments.contains("-UITestDisableCountdown")
        let countdownDuration = disableCountdown ? 0 : 3
        #else
        let countdownDuration = 3
        #endif

        return RecordingViewModel(
            startRecordingUseCase: startRecordingUseCase,
            stopRecordingUseCase: stopRecordingUseCase,
            audioPlayer: audioPlayer,
            pitchDetector: pitchDetector,
            subscriptionViewModel: subscriptionViewModel,
            countdownDuration: countdownDuration
        )
    }()

    // Subscription ViewModels
    public lazy var subscriptionViewModel: SubscriptionViewModel = {
        SubscriptionViewModel(
            getStatusUseCase: getSubscriptionStatusUseCase,
            purchaseUseCase: purchaseSubscriptionUseCase,
            restoreUseCase: restorePurchasesUseCase
        )
    }()

    public lazy var paywallViewModel: PaywallViewModel = {
        PaywallViewModel(
            getStatusUseCase: getSubscriptionStatusUseCase,
            purchaseUseCase: purchaseSubscriptionUseCase,
            restoreUseCase: restorePurchasesUseCase,
            getProductsUseCase: getAvailableProductsUseCase
        )
    }()

    // Audio Settings ViewModel Factories
    func makeAudioInputSettingsViewModel() -> AudioInputSettingsViewModel {
        AudioInputSettingsViewModel(repository: audioSettingsRepository)
    }

    func makeAudioOutputSettingsViewModel() -> AudioOutputSettingsViewModel {
        AudioOutputSettingsViewModel(repository: audioSettingsRepository)
    }

    func makeAlgorithmSettingsViewModel() -> AlgorithmSettingsViewModel {
        AlgorithmSettingsViewModel(repository: audioSettingsRepository)
    }

    // MARK: - Setup

    private func setupInfrastructure() {
        // Configure audio session if needed
    }

    private func setupUseCases() {
        // Initialize use cases
    }
}
