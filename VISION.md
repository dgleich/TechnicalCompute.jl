The vision here is to make it easy to include a ton of functionality in Julia.

1. A very rich compute environment by default with the most commonly used packages auto loaded.
   As a rough guide, it would be nice to be able to do the entire MIT Comp. Thinking class
   without any additional packages, or with only very small exceptions.

2. A place to put 'local codes' that will be automatically included in Julia. These are routines
   you might like and use frequently.

3. Ability to select a subset of functionality that you'd like to include to keep things faster.

   ```
   julia> technical_compute_setup()
   ```

   would launch a routine to customize which packages are included.

   @technical_compute_add_local, this will add your local package/directory.

---

As a guide, the idea is as a comparison to Matlab/Scipy. We don't want users wading through
tons of packages to figure out what to include. It should all be there. Experts can downselect
if they need to. TechnicalCompute is designed to be big and expressive. 

---


