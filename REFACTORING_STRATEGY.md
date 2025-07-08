Of course. I have analyzed the `data-viewer` application and create a refactoring plan to improve its readability, maintainability, and efficiency.

Here is the proposed refactoring strategy. I will also save this as `REFACTORING_STRATEGY.md` in the project root.

# Refactoring Strategy for Data Viewer

This document outlines a plan to refactor the `data-viewer` application. The primary goals are to improve code structure, readability, and maintainability without altering the core functionality.

### 1. Project Structure & Organization

The current structure is flat. A more modular approach will make the codebase easier to navigate.

-   **Create new directories** within `src/` for better organization:
    -   `src/hooks/`: For custom React hooks (e.g., `useImageFilters`, `useCameraControls`).
    -   `src/store/`: For Zustand state management stores.
    -   `src/types/`: For shared TypeScript type definitions and interfaces.
    -   `src/utils/`: Consolidate all utility functions here.
-   **Group components** by feature:
    -   `src/components/viewer/`: For components related to the main 3D scene (`Scene`, `File`, `Title3D`, etc.).
    -   `src/components/ui/`: For UI elements like `FilterSidebar`, `ImageViewer`, `LoadingAnimation`.

### 2. State Management

Currently, state is managed primarily within the `App` component using `useState`. This can lead to prop drilling and makes state difficult to track.

-   **Introduce a centralized state manager:** Use **Zustand** (which is already a dependency) to create a global store.
-   **Consolidate state:** Move the following states from `App.tsx` into a new `src/store/viewerStore.ts`:
    -   `filters`
    -   `selectedImage` & `selectedImageData`
    -   `isLoading`
    -   `isFilterSidebarOpen`
-   **Refactor components** to use the Zustand store instead of receiving props for state and callbacks. This will simplify `App.tsx` significantly.

### 3. Data Handling and Services

Image data and S3 logic are spread across a few files. This can be centralized.

-   **Unify Data Source:** The `s3-actual-images.ts` and `s3-image-list.ts` files are confusing.
    -   Remove the mock data generation in `s3-image-list.ts`.
    -   Standardize on using `s3-actual-images.ts` as the single source of truth for image paths, which can be updated by the `generate-image-list.js` script.
-   **Create a Data Utility Module:**
    -   Move the `imageData` parsing logic and the `findImagePair` function from `Scene.tsx` into a new `src/utils/imageDataUtils.ts`. This will decouple the `Scene` component from the data transformation logic.
-   **Refactor Image Loader:**
    -   The `ProgressiveImageLoader` in `src/utils/imageLoader.ts` should be instantiated once and provided via a React Context or the Zustand store to avoid creating multiple instances.

### 4. Component Refactoring

-   **`App.tsx`:** This component should be simplified to act as a layout and composition root.
    -   The filter logic will be handled by the Zustand store.
    -   The main JSX can be cleaned up, with less prop passing.
-   **`Scene.tsx`:**
    -   Remove data parsing logic (moved to `imageDataUtils.ts`).
    -   Get filtered images from a selector on the Zustand store.
    -   The `BatchLoadedImages` internal component can be extracted into its own file within `src/components/viewer/`.
-   **`FreeCameraControls.tsx`:** This component is complex.
    -   Add detailed comments to explain the momentum, damping, and motion blur logic.
    -   Break down the large `useEffect` into smaller, more focused effects for different event listeners (e.g., `useMouse`, `useWheel`).
    -   Extract the complex mathematical calculations into pure utility functions.

### 5. Styling

-   **Consolidate Styles:** Move the inline styles used for the "Filters" button in `App.tsx` into the `App.css` file to maintain a clean separation of concerns.
-   **Use CSS Variables:** Continue to leverage the CSS variables defined in `index.css` for consistency.

### 6. Scripts and Configuration

-   **Update Scripts:** The shell scripts in the `scripts/` directory contain placeholder values like `<your-bucket-name>`. These should be updated to use environment variables for easier configuration.
-   **Clarify Data Scripts:** Rename `generate-image-list.js` to `generate-s3-file-list.js` to match its purpose and update its output to `s3-actual-images.ts` to avoid confusion.

By following this plan, the `data-viewer` application will become more modular, scalable, and easier for developers to understand and maintain.